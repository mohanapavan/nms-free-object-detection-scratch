import os
import random
from pathlib import Path
import cv2
import numpy
import torch
from torch.utils import data

FORMATS = {"bmp", "jpeg", "jpg", "png", "tif", "tiff", "webp"}

def resample():
    return random.choice((cv2.INTER_AREA, cv2.INTER_CUBIC,
                           cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4))

def augment_hsv(image, params):
    r  = numpy.random.uniform(-1, 1, 3) * [params["hsv_h"], params["hsv_s"], params["hsv_v"]] + 1
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    x = numpy.arange(0, 256, dtype=r.dtype)
    cv2.cvtColor(cv2.merge((cv2.LUT(h, ((x*r[0])%180).astype("uint8")),
                             cv2.LUT(s, numpy.clip(x*r[1],0,255).astype("uint8")),
                             cv2.LUT(v, numpy.clip(x*r[2],0,255).astype("uint8")))),
                 cv2.COLOR_HSV2BGR, dst=image)

def resize_img(image, input_size, augment):
    shape  = image.shape[:2]
    r      = min(input_size / shape[0], input_size / shape[1])
    if not augment: r = min(r, 1.0)
    new_wh = int(round(shape[1]*r)), int(round(shape[0]*r))
    w = (input_size - new_wh[0]) / 2
    h = (input_size - new_wh[1]) / 2
    if shape[::-1] != new_wh:
        image = cv2.resize(image, new_wh, interpolation=resample() if augment else cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image,
                                int(round(h-0.1)), int(round(h+0.1)),
                                int(round(w-0.1)), int(round(w+0.1)),
                                cv2.BORDER_CONSTANT)
    return image, (r, r), (w, h)

def wh2xy_np(x, w=640, h=640, pad_w=0, pad_h=0):
    y = numpy.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2]/2) + pad_w
    y[:, 1] = h * (x[:, 1] - x[:, 3]/2) + pad_h
    y[:, 2] = w * (x[:, 0] + x[:, 2]/2) + pad_w
    y[:, 3] = h * (x[:, 1] + x[:, 3]/2) + pad_h
    return y

def xy2wh_np(x, w, h):
    x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1e-3)
    x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1e-3)
    y = numpy.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h
    y[:, 2] = (x[:, 2] - x[:, 0]) / w
    y[:, 3] = (x[:, 3] - x[:, 1]) / h
    return y

def random_perspective(image, label, params, border=(0, 0)):
    h = image.shape[0] + border[0] * 2
    w = image.shape[1] + border[1] * 2
    C = numpy.eye(3); C[0,2] = -image.shape[1]/2; C[1,2] = -image.shape[0]/2
    P = numpy.eye(3)
    P[2,0] = random.uniform(-params["perspective"], params["perspective"])
    P[2,1] = random.uniform(-params["perspective"], params["perspective"])
    R = numpy.eye(3)
    a = random.uniform(-params["degrees"], params["degrees"])
    s = random.uniform(1 - params["scale"], 1 + params["scale"])
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0,0), scale=s)
    S = numpy.eye(3)
    S[0,1] = math.tan(random.uniform(-params["shear"], params["shear"]) * math.pi / 180)
    S[1,0] = math.tan(random.uniform(-params["shear"], params["shear"]) * math.pi / 180)
    T = numpy.eye(3)
    T[0,2] = random.uniform(0.5-params["translate"], 0.5+params["translate"]) * w
    T[1,2] = random.uniform(0.5-params["translate"], 0.5+params["translate"]) * h
    M = T @ S @ R @ P @ C
    if (border[0] != 0) or (border[1] != 0) or (M != numpy.eye(3)).any():
        image = cv2.warpAffine(image, M[:2], dsize=(w, h), borderValue=(0,0,0))
    n = len(label)
    if n:
        xy = numpy.ones((n*4, 3))
        xy[:, :2] = label[:, [1,2,3,4,1,4,3,2]].reshape(n*4, 2)
        xy = (xy @ M.T)[:, :2].reshape(n, 8)
        x  = xy[:, [0,2,4,6]]; y = xy[:, [1,3,5,7]]
        new = numpy.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        new[:, [0,2]] = new[:, [0,2]].clip(0, w)
        new[:, [1,3]] = new[:, [1,3]].clip(0, h)
        w1,h1 = label[:,3]-label[:,1], label[:,4]-label[:,2]
        w2,h2 = new[:,2]-new[:,0], new[:,3]-new[:,1]
        ok = (w2>2)&(h2>2)&(w2*h2/(w1*h1+1e-16)>0.1)
        label = label[ok]; label[:,1:5] = new[ok]
    return image, label

def mix_up(image1, label1, image2, label2):
    alpha = numpy.random.beta(32.0, 32.0)
    return (image1 * alpha + image2 * (1 - alpha)).astype(numpy.uint8), \
           numpy.concatenate((label1, label2), 0)

class SimpleDataset(data.Dataset):
    def __init__(self, img_dir, input_size, params, augment=True):
        self.params     = params
        self.augment    = augment
        self.mosaic     = augment
        self.input_size = input_size
        exts = {".jpg",".jpeg",".png",".bmp",".tiff",".webp"}
        self.filenames = [str(p) for p in sorted(Path(img_dir).rglob("*")) if p.suffix.lower() in exts]
        assert self.filenames, f"No images found in {img_dir}"
        self.labels = []
        for fp in self.filenames:
            lp = Path(fp.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")).with_suffix(".txt")
            if lp.is_file():
                rows = [r.split() for r in lp.read_text().strip().splitlines() if r]
                lbl  = numpy.array(rows, dtype=numpy.float32) if rows else numpy.zeros((0,5), numpy.float32)
            else:
                lbl = numpy.zeros((0,5), numpy.float32)
            self.labels.append(lbl)
        self.indices = list(range(len(self.filenames)))

    def __len__(self): return len(self.filenames)

    def load_image(self, i):
        img = cv2.imread(self.filenames[i])
        h, w = img.shape[:2]
        r = self.input_size / max(h, w)
        if r != 1:
            img = cv2.resize(img, (int(w*r), int(h*r)),
                             interpolation=resample() if self.augment else cv2.INTER_LINEAR)
        return img, (h, w)

    def load_mosaic(self, index):
        label4 = []; s = self.input_size
        image4 = numpy.full((s*2, s*2, 3), 0, dtype=numpy.uint8)
        border = [-s//2, -s//2]
        xc = int(random.uniform(-border[0], 2*s+border[1]))
        yc = int(random.uniform(-border[0], 2*s+border[1]))
        for i, idx in enumerate(random.choices(self.indices, k=3) + [index]):
            img, _ = self.load_image(idx)
            sh = img.shape
            if i == 0: x1a,y1a,x2a,y2a=max(xc-sh[1],0),max(yc-sh[0],0),xc,yc; x1b,y1b,x2b,y2b=sh[1]-(x2a-x1a),sh[0]-(y2a-y1a),sh[1],sh[0]
            elif i==1: x1a,y1a,x2a,y2a=xc,max(yc-sh[0],0),min(xc+sh[1],s*2),yc; x1b,y1b,x2b,y2b=0,sh[0]-(y2a-y1a),min(sh[1],x2a-x1a),sh[0]
            elif i==2: x1a,y1a,x2a,y2a=max(xc-sh[1],0),yc,xc,min(s*2,yc+sh[0]); x1b,y1b,x2b,y2b=sh[1]-(x2a-x1a),0,sh[1],min(y2a-y1a,sh[0])
            else:      x1a,y1a,x2a,y2a=xc,yc,min(xc+sh[1],s*2),min(s*2,yc+sh[0]); x1b,y1b,x2b,y2b=0,0,min(sh[1],x2a-x1a),min(y2a-y1a,sh[0])
            image4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            lbl = self.labels[idx].copy()
            if len(lbl): lbl[:,1:] = wh2xy_np(lbl[:,1:], sh[1], sh[0], x1a-x1b, y1a-y1b)
            label4.append(lbl)
        label4 = numpy.concatenate(label4, 0)
        numpy.clip(label4[:,1:], 0, 2*s, out=label4[:,1:])
        return random_perspective(image4, label4, self.params, border)

    def __getitem__(self, index):
        params = self.params
        if self.mosaic and random.random() < params["mosaic"]:
            image, label = self.load_mosaic(index)
            if random.random() < params["mix_up"]:
                img2, lbl2 = self.load_mosaic(random.choice(self.indices))
                image, label = mix_up(image, label, img2, lbl2)
        else:
            image, shape = self.load_image(index)
            h, w  = image.shape[:2]
            image, ratio, pad = resize_img(image, self.input_size, self.augment)
            label = self.labels[index].copy()
            if label.size:
                label[:,1:] = wh2xy_np(label[:,1:], ratio[0]*w, ratio[1]*h, pad[0], pad[1])
            if self.augment:
                image, label = random_perspective(image, label, params)

        h_img, w_img = image.shape[:2]
        cls = label[:, 0:1]
        box = label[:, 1:5]
        box = xy2wh_np(box, w_img, h_img) if len(box) else box

        if self.augment:
            augment_hsv(image, params)
            nl = len(box)
            if random.random() < params["flip_ud"]:
                image = numpy.flipud(image)
                if nl: box[:,1] = 1 - box[:,1]
            if random.random() < params["flip_lr"]:
                image = numpy.fliplr(image)
                if nl: box[:,0] = 1 - box[:,0]

        nl = len(box)
        target_cls = torch.from_numpy(cls.reshape(nl,1).copy()) if nl else torch.zeros((0,1))
        target_box = torch.from_numpy(box.copy())               if nl else torch.zeros((0,4))

        sample = numpy.ascontiguousarray(image.transpose((2,0,1))[::-1])
        return torch.from_numpy(sample), target_cls, target_box, torch.zeros(nl)

    @staticmethod
    def collate_fn(batch):
        samples, cls, box, indices = zip(*batch)
        cls = torch.cat(cls, dim=0)
        box = torch.cat(box, dim=0)
        new_indices = list(indices)
        for i in range(len(indices)):
            new_indices[i] = new_indices[i] + i
        indices = torch.cat(new_indices, dim=0)
        return torch.stack(samples, 0), {"cls": cls, "box": box, "idx": indices}