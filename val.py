import torch
import numpy
from torch.utils import data
from dataset import SimpleDataset
from utils import wh2xy_torch
from config import PARAMS

# Calculate evaluation metric
def compute_metric(output, target, iou_v):
    a1, a2 = target[:, 1:].unsqueeze(1).chunk(2, 2)
    b1, b2 = output[:, :4].unsqueeze(0).chunk(2, 2)
    inter  = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    iou    = inter / ((a2-a1).prod(2) + (b2-b1).prod(2) - inter + 1e-7)
    correct = numpy.zeros((output.shape[0], iou_v.shape[0]), dtype=bool)
    for i in range(len(iou_v)):
        x = torch.where((iou >= iou_v[i]) & (target[:,0:1] == output[:,5]))
        if x[0].shape[0]:
            m = torch.cat((torch.stack(x,1), iou[x[0],x[1]][:,None]),1).cpu().numpy()
            if x[0].shape[0] > 1:
                m = m[m[:,2].argsort()[::-1]]
                m = m[numpy.unique(m[:,1], return_index=True)[1]]
                m = m[numpy.unique(m[:,0], return_index=True)[1]]
            correct[m[:,1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=output.device)

# Main validation pass
@torch.no_grad()
def validate(model, val_dir, input_size, batch_size=16):
    model.eval()
    dataset = SimpleDataset(val_dir, input_size, PARAMS, augment=False)
    loader  = data.DataLoader(dataset, batch_size, shuffle=False,
                               num_workers=4, pin_memory=True,
                               collate_fn=SimpleDataset.collate_fn)
    iou_v   = torch.linspace(0.5, 0.95, 10).cuda()
    metrics = []

    for samples, targets in loader:
        samples = samples.cuda().float() / 255
        _, _, H, W = samples.shape
        scale   = torch.tensor((W, H, W, H)).cuda()
        outputs = model(samples)                       
        outputs = [outputs[b][outputs[b,:,4] > 0.001] for b in range(outputs.shape[0])]

        for i, output in enumerate(outputs):
            idx = targets["idx"] == i
            cls = targets["cls"][idx].cuda()
            box = targets["box"][idx].cuda()
            metric = torch.zeros(output.shape[0], len(iou_v), dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if cls.shape[0]:
                    metrics.append((metric, *torch.zeros((2,0)).cuda(), cls.squeeze(-1)))
                continue
            if cls.shape[0]:
                gt_xyxy = wh2xy_torch(box) * scale
                target_ = torch.cat((cls, gt_xyxy), dim=1)
                metric  = compute_metric(output[:, :6], target_, iou_v)
            metrics.append((metric, output[:,4], output[:,5], cls.squeeze(-1)))

    if not metrics:
        model.float(); model.train(); return 0.0

    tp_all, conf_all, cls_all, tcls_all = [torch.cat(x,0).cpu().numpy() for x in zip(*metrics)]
    eps = 1e-16
    ap50_list = []
    for c in numpy.unique(tcls_all):
        mask   = cls_all == c
        n_true = (tcls_all == c).sum()
        if not mask.sum() or not n_true: continue
        order  = numpy.argsort(-conf_all[mask])
        tp_c   = tp_all[mask][order, 0]
        fpc    = (1 - tp_c).cumsum()
        tpc    = tp_c.cumsum()
        rec    = tpc / (n_true + eps)
        pre    = tpc / (tpc + fpc + eps)
        m_rec  = numpy.concatenate(([0.], rec, [1.]))
        m_pre  = numpy.concatenate(([1.], pre, [0.]))
        m_pre  = numpy.flip(numpy.maximum.accumulate(numpy.flip(m_pre)))
        x      = numpy.linspace(0, 1, 101)
        ap50_list.append(numpy.trapz(numpy.interp(x, m_rec, m_pre), x))

    map50 = float(numpy.mean(ap50_list)) if ap50_list else 0.0
    model.float(); model.train()
    return map50