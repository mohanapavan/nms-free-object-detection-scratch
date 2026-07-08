import torch
from utils import compute_iou, xy2wh_torch, wh2xy_torch, make_anchors

class BoxLoss(torch.nn.Module):
    @staticmethod
    def forward(pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum,
                fg_mask, size, stride):
        weight   = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou      = compute_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        x1y1, x2y2    = target_bboxes.chunk(2, -1)
        target_ltrb   = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1) * stride
        target_ltrb[..., 0::2] /= size[1]
        target_ltrb[..., 1::2] /= size[0]
        pred_d = pred_dist * stride
        pred_d[..., 0::2] /= size[1]
        pred_d[..., 1::2] /= size[0]
        loss_l1 = (torch.nn.functional.l1_loss(pred_d[fg_mask], target_ltrb[fg_mask], reduction="none")
                   .mean(-1, keepdim=True) * weight).sum() / target_scores_sum
        return loss_iou, loss_l1

class Assigner(torch.nn.Module):
    def __init__(self, top_k1, top_k2, nc, stride):
        super().__init__()
        self.nc     = nc
        self.top_k1 = top_k1
        self.top_k2 = top_k2 or top_k1
        self.stride = stride
        self.alpha  = 0.5
        self.beta   = 6
        self.eps    = 1e-9

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        batch_size    = pd_scores.shape[0]
        n_max_boxes   = gt_bboxes.shape[1]

        if n_max_boxes == 0:
            return (torch.zeros_like(pd_bboxes),
                    torch.zeros_like(pd_scores),
                    torch.zeros_like(pd_scores[..., 0]),
                    torch.zeros_like(pd_scores[..., 0]))

        gt_bboxes_xywh = xy2wh_torch(gt_bboxes)
        wh_mask        = gt_bboxes_xywh[..., 2:] < self.stride[0]
        stride_val     = torch.tensor(self.stride[1], dtype=gt_bboxes_xywh.dtype, device=gt_bboxes_xywh.device)
        gt_bboxes_xywh[..., 2:] = torch.where((wh_mask * mask_gt).bool(), stride_val, gt_bboxes_xywh[..., 2:])
        gt_bboxes_xy   = wh2xy_torch(gt_bboxes_xywh)

        na     = pd_bboxes.shape[-2]
        bs, n_boxes, _ = gt_bboxes_xy.shape
        lt, rb = gt_bboxes_xy.view(-1, 1, 4).chunk(2, 2)
        bbox_deltas = torch.cat((anc_points[None] - lt, rb - anc_points[None]), dim=2).view(bs, n_boxes, na, -1)
        mask_in_gts = bbox_deltas.amin(3).gt_(1e-9)

        mask     = (mask_in_gts * mask_gt).bool()
        overlaps = torch.zeros([batch_size, n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([batch_size, n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind    = torch.zeros([2, batch_size, n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(batch_size).view(-1, 1).expand(-1, n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores[mask] = pd_scores[ind[0], :, ind[1]][mask]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, n_max_boxes, -1, -1)[mask]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask]
        overlaps[mask] = compute_iou(gt_boxes, pd_boxes).squeeze(-1).clamp_(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)

        top_k_mask              = mask_gt.expand(-1, -1, self.top_k1).bool()
        top_k_metrics, top_k_indices = torch.topk(align_metric, self.top_k1, dim=-1, largest=True)
        top_k_indices.masked_fill_(~top_k_mask, 0)

        count_tensor = torch.zeros(align_metric.shape, dtype=torch.int8, device=top_k_indices.device)
        ones         = torch.ones_like(top_k_indices[:, :, :1], dtype=torch.int8, device=top_k_indices.device)
        for k in range(self.top_k1):
            count_tensor.scatter_add_(-1, top_k_indices[:, :, k:k+1], ones)
        count_tensor.masked_fill_(count_tensor > 1, 0)

        mask_pos = count_tensor.to(align_metric.dtype) * mask_in_gts * mask_gt
        fg_mask  = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi_gts   = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
            max_overlaps_idx = overlaps.argmax(1)
            is_max_overlaps  = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            fg_mask = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float().sum(-2)

        if self.top_k2 != self.top_k1:
            max_overlaps_idx = torch.topk(align_metric * mask_pos, self.top_k2, dim=-1, largest=True).indices
            topk_idx = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            topk_idx.scatter_(-1, max_overlaps_idx, 1.0)
            fg_mask = (mask_pos * topk_idx).sum(-2)

        target_gt_idx  = mask_pos.argmax(-2)
        batch_ind      = torch.arange(batch_size, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_idx     = target_gt_idx + batch_ind * n_max_boxes
        target_labels  = gt_labels.long().flatten()[target_idx]
        target_bboxes  = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_idx]
        target_labels.clamp_(0)

        target_scores = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                     dtype=torch.int64, device=target_labels.device)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)
        target_scores  = torch.where(fg_scores_mask > 0, target_scores, 0)

        align_metric      *= mask_pos
        pos_align_metrics  = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps       = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric  = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores      = target_scores * norm_align_metric

        return target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

class SingleLoss:
    def __init__(self, model, params, top_k1=10, top_k2=None):
        self.nc      = model.head.nc
        self.stride  = model.head.stride
        self.params  = params
        self.device  = next(model.parameters()).device
        self.box_loss = BoxLoss().to(self.device)
        self.cls_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = Assigner(top_k1, top_k2, self.nc, self.stride.tolist())

    def __call__(self, outputs, targets):
        loss = torch.zeros(3, device=self.device)
        pred_dist, pred_scores = (outputs["boxes"].permute(0, 2, 1).contiguous(),
                                  outputs["scores"].permute(0, 2, 1).contiguous())
        anchor_points, stride_tensor = make_anchors(outputs["x"], self.stride)

        dtype      = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        size       = torch.tensor(outputs["x"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        t = torch.cat((targets["idx"].view(-1, 1), targets["cls"].view(-1, 1), targets["box"]), 1).to(self.device)
        nl, ne = t.shape
        if nl == 0:
            gt = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = t[:, 0]
            _, counts = i.unique(return_counts=True)
            gt = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                m = i == j
                if n := m.sum():
                    gt[j, :n] = t[m, 1:]
            gt[..., 1:5] = wh2xy_torch(gt[..., 1:5].mul_(size[[1, 0, 1, 0]]))

        gt_labels, gt_bboxes = gt.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        lt, rb = pred_dist.chunk(2, -1)
        pred_bboxes = torch.cat((anchor_points - lt, anchor_points + rb), -1)

        assigned = self.assigner(pred_scores.detach().sigmoid(),
                                 (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                                 anchor_points * stride_tensor,
                                 gt_labels, gt_bboxes, mask_gt)
        target_bboxes, target_scores, fg_mask, _ = assigned

        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.cls_loss(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[2] = self.box_loss(pred_dist, pred_bboxes, anchor_points,
                                              target_bboxes / stride_tensor,
                                              target_scores, target_scores_sum,
                                              fg_mask, size, stride_tensor)
        loss[0] *= self.params["box"]
        loss[1] *= self.params["cls"]
        loss[2] *= self.params["dfl"]
        return loss

class ComputeLoss:
    def __init__(self, model, params, epochs):
        self.epochs  = epochs
        self.loss_o2m = SingleLoss(model, params, top_k1=10)         
        self.loss_o2o = SingleLoss(model, params, top_k1=7, top_k2=1) 
        self.a = 0.8   
        self.updates = 0

    def __call__(self, outputs, targets):
        o2m_out, o2o_out = outputs          
        l1 = self.loss_o2m(o2m_out, targets)
        l2 = self.loss_o2o(o2o_out, targets)
        c  = max(1.0 - self.a, 0.0)
        return (self.a * l1 + c * l2).sum()

    def step(self):
        self.updates += 1
        self.a = max(1 - self.updates / max(self.epochs - 1, 1), 0) * (0.8 - 0.1) + 0.1