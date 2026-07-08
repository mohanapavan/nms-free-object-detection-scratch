import copy
import math
import torch
from utils import make_anchors

class Conv(torch.nn.Module):
    def __init__(self, c1, c2, relu, k=1, s=1, p=0, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.norm = torch.nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.relu = relu

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class Residual(torch.nn.Module):
    def __init__(self, in_ch, out_ch, add=True, e=0.5):
        super().__init__()
        self.add_m = add and in_ch == out_ch
        self.conv1 = Conv(in_ch, int(out_ch * e), torch.nn.SiLU(), k=3, s=1, p=1)
        self.conv2 = Conv(int(out_ch * e), out_ch, torch.nn.SiLU(), k=3, s=1, p=1)

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.add_m else y

class Attention(torch.nn.Module):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.key_dim   = int(self.head_dim * attn_ratio)
        self.scale     = self.key_dim ** -0.5
        self.qkv  = Conv(dim, dim + (self.key_dim * num_heads) * 2, torch.nn.Identity())
        self.proj = Conv(dim, dim, torch.nn.Identity())
        self.pe   = Conv(dim, dim, torch.nn.Identity(), 3, 1, 1, dim)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x).view(b, self.num_heads, self.key_dim * 2 + self.head_dim, h * w)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
        return self.proj(x)

class PSABlock(torch.nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, add=True):
        super().__init__()
        self.add_m = add
        self.conv1 = Attention(c, num_heads, attn_ratio)
        self.conv2 = torch.nn.Sequential(Conv(c, c * 2, torch.nn.SiLU()),
                                         Conv(c * 2, c, torch.nn.Identity()))

    def forward(self, x):
        x = x + self.conv1(x) if self.add_m else self.conv1(x)
        x = x + self.conv2(x) if self.add_m else self.conv2(x)
        return x

class CSPModule(torch.nn.Module):
    def __init__(self, in_ch, out_ch, add=True, e=0.5):
        super().__init__()
        self.conv1 = Conv(in_ch, int(out_ch * e), torch.nn.SiLU())
        self.conv2 = Conv(in_ch, int(out_ch * e), torch.nn.SiLU())
        self.conv3 = Conv(2 * int(out_ch * e), out_ch, torch.nn.SiLU())
        self.res_m = torch.nn.Sequential(
            Residual(int(out_ch * e), int(out_ch * e), add, e=1.0),
            Residual(int(out_ch * e), int(out_ch * e), add, e=1.0))

    def forward(self, x):
        return self.conv3(torch.cat((self.res_m(self.conv1(x)), self.conv2(x)), dim=1))

class CSP(torch.nn.Module):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, add=True):
        super().__init__()
        self.conv1 = Conv(c1, 2 * int(c2 * e), torch.nn.SiLU())
        self.conv2 = Conv((2 + n) * int(c2 * e), c2, torch.nn.SiLU())
        modules = []
        for _ in range(n):
            if c3k:
                if attn:
                    modules.append(torch.nn.Sequential(
                        Residual(int(c2 * e), int(c2 * e), add),
                        PSABlock(int(c2 * e), num_heads=max(int(c2 * e) // 64, 1))))
                else:
                    modules.append(CSPModule(int(c2 * e), int(c2 * e), add))
            else:
                modules.append(Residual(int(c2 * e), int(c2 * e), add))
        self.res_m = torch.nn.ModuleList(modules)

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv2(torch.cat(y, dim=1))

class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5, n=3, add=True):
        super().__init__()
        self.n = n
        self.add_m = add and in_ch == out_ch
        self.conv1 = Conv(in_ch, in_ch // 2, torch.nn.Identity())
        self.conv2 = Conv((in_ch // 2) * (n + 1), out_ch, torch.nn.SiLU())
        self.res_m = torch.nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        y = [self.conv1(x)]
        y.extend(self.res_m(y[-1]) for _ in range(self.n))
        y = self.conv2(torch.cat(y, dim=1))
        return x + y if self.add_m else y

class PSA(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, e=0.5):
        super().__init__()
        assert in_ch == out_ch
        self.c = int(in_ch * e)
        self.conv1 = Conv(in_ch, 2 * self.c, torch.nn.SiLU())
        self.conv2 = Conv(2 * self.c, in_ch, torch.nn.SiLU())
        self.res_m = torch.nn.Sequential(*(PSABlock(self.c, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        a, b = self.conv1(x).split((self.c, self.c), dim=1)
        return self.conv2(torch.cat((a, self.res_m(b)), dim=1))

class Backbone(torch.nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()
        self.p1 = torch.nn.Sequential(Conv(width[0], width[1], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p2 = torch.nn.Sequential(Conv(width[1], width[2], torch.nn.SiLU(), k=3, s=2, p=1),
                                       CSP(width[2], width[3], depth[0], csp[0], e=0.25))
        self.p3 = torch.nn.Sequential(Conv(width[3], width[3], torch.nn.SiLU(), k=3, s=2, p=1),
                                       CSP(width[3], width[4], depth[1], csp[0], e=0.25))
        self.p4 = torch.nn.Sequential(Conv(width[4], width[4], torch.nn.SiLU(), k=3, s=2, p=1),
                                       CSP(width[4], width[4], depth[2], csp[1]))
        self.p5 = torch.nn.Sequential(Conv(width[4], width[5], torch.nn.SiLU(), k=3, s=2, p=1),
                                       CSP(width[5], width[5], depth[3], csp[1]),
                                       SPP(width[5], width[5]),
                                       PSA(width[5], width[5], depth[4]))

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return [p3, p4, p5]

class Neck(torch.nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[5], csp[1])
        self.h2 = CSP(width[4] + width[4], width[3], depth[5], csp[1])
        self.h3 = Conv(width[3], width[3], torch.nn.SiLU(), k=3, s=2, p=1)
        self.h4 = CSP(width[3] + width[4], width[4], depth[5], csp[1])
        self.h5 = Conv(width[4], width[4], torch.nn.SiLU(), k=3, s=2, p=1)
        self.h6 = CSP(width[4] + width[5], width[5], depth[5], csp[1], attn=True)

    def forward(self, x):
        p3, p4, p5 = x
        p4 = self.h1(torch.cat([self.up(p5), p4], dim=1))
        p3 = self.h2(torch.cat([self.up(p4), p3], dim=1))
        p4 = self.h4(torch.cat([self.h3(p3), p4], dim=1))
        p5 = self.h6(torch.cat([self.h5(p4), p5], dim=1))
        return [p3, p4, p5]

class Head(torch.nn.Module):
    shape   = None
    max_det = 300
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc, filters):
        super().__init__()
        self.nc     = nc
        self.no     = nc + 4
        self.nl     = len(filters)
        self.stride = torch.zeros(self.nl)

        box = max(16, filters[0] // 4)
        cls = max(filters[0], max(min(nc, 100), 80))

        self.box_train = torch.nn.ModuleList(
            torch.nn.Sequential(Conv(f, box, torch.nn.SiLU(), 3, p=1),
                                 Conv(box, box, torch.nn.SiLU(), 3, p=1),
                                 torch.nn.Conv2d(box, 4, 1)) for f in filters)
        self.cls_train = torch.nn.ModuleList(
            torch.nn.Sequential(Conv(f, f,   torch.nn.SiLU(), k=3, p=1, g=f),
                                 Conv(f, cls, torch.nn.SiLU()),
                                 Conv(cls, cls, torch.nn.SiLU(), k=3, p=1, g=cls),
                                 Conv(cls, cls, torch.nn.SiLU()),
                                 torch.nn.Conv2d(cls, nc, 1)) for f in filters)

        self.box_head = copy.deepcopy(self.box_train)
        self.cls_head = copy.deepcopy(self.cls_train)

    def _forward(self, x, box_head, cls_head):
        bs    = x[0].shape[0]
        boxes  = torch.cat([box_head[i](x[i]).view(bs, 4, -1)       for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(x=x, boxes=boxes, scores=scores)

    def forward(self, x):
        if self.training:
            y1 = self._forward(x, self.box_train, self.cls_train) 
            y2 = self._forward([i.detach() for i in x], self.box_head, self.cls_head)
            return y1, y2

        x = [i.detach() for i in x]
        y = self._forward(x, self.box_head, self.cls_head)
        shape = y["x"][0].shape
        if self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(y["x"], self.stride))
            self.shape = shape

        box     = y["boxes"]
        anchors = self.anchors.unsqueeze(0)
        lt, rb  = box.chunk(2, 1)
        box     = torch.cat((anchors - lt, anchors + rb), 1) * self.strides
        out     = torch.cat((box, y["scores"].sigmoid()), dim=1).permute(0, 2, 1)

        boxes_out, scores_out = out.split([4, self.nc], dim=-1)
        bs, n_anchors, nc = scores_out.shape
        k         = min(self.max_det, n_anchors)
        ori_index = scores_out.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
        scores_   = scores_out.gather(dim=1, index=ori_index.repeat(1, 1, nc))
        scores_, index = scores_.flatten(1).topk(k)
        idx    = ori_index[torch.arange(bs)[..., None], index // nc]
        conf   = (index % nc)[..., None].float()
        boxes_ = boxes_out.gather(dim=1, index=idx.repeat(1, 1, 4))
        return torch.cat([boxes_, scores_[..., None], conf], dim=-1)

    def initialize_biases(self):
        for i, (a, b) in enumerate(zip(self.box_train, self.cls_train)):
            a[-1].bias.data[:] = 2.0
            b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        for i, (a, b) in enumerate(zip(self.box_head, self.cls_head)):
            a[-1].bias.data[:] = 2.0
            b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)

class YOLO(torch.nn.Module):
    def __init__(self, width, depth, csp, num_classes):
        super().__init__()
        self.backbone = Backbone(width, depth, csp)
        self.neck     = Neck(width, depth, csp)
        self.head     = Head(num_classes, (width[3], width[4], width[5]))
        dummy = torch.zeros(1, width[0], 256, 256)
        self.head.stride = torch.tensor([256 / i.shape[-2] for i in self.forward(dummy)[0]["x"]])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return self.head(x)

def yolo_v26_n(num_classes):
    return YOLO([3, 16, 32, 64, 128, 256], [1,1,1,1,1,1], [False, True], num_classes)