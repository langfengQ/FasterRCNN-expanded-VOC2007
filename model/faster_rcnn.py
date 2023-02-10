import torch.nn as nn
from utils.config import config
import torch
import numpy as np
from model.bbox_tools import *
import torch.nn.functional as F
from torchvision.ops import nms


def nograd(f):
    def new_f(*args, **kwargs):
        with torch.no_grad():
            return f(*args, **kwargs)
    return new_f


class FasterRCNN(nn.Module):
    def __init__(self,
                 extractor,
                 rpn,
                 head,
                 offset_normalize_mean=(0., 0., 0., 0.),
                 offset_normalize_std=(0.1, 0.1, 0.2, 0.2),
                 ):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head
        self.offset_normalize_mean = offset_normalize_mean
        self.offset_normalize_std = offset_normalize_std

    @property
    def n_class(self):
        return self.head.n_class

    def forward(self, x, scale):
        img_size = x.shape[2:]
        x = self.extractor(x)
        rois, rois_index, _, _, _ = self.rpn(
            x=x, img_size=img_size, scale=scale)
        roi_bbox_pred, roi_cls_scores = self.head(
            x=x, rois=rois, rois_index=rois_index)
        return rois, roi_bbox_pred, roi_cls_scores

    @nograd
    def predict(self, img, scale=1., eval=True):
        if eval:
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
            # self.score_thresh = 0.65
        else:
            self.nms_thresh = 0.3
            self.score_thresh = 0.7

        _, _, H, W = img.shape
        img_size = (H, W)
        device = img.device

        self.eval()
        # with torch.no_grad():
        roi, roi_bbox_pred, roi_cls_scores = self(img, scale=scale)

        mean = torch.Tensor(self.offset_normalize_mean).to(
            device)[None, None, :]
        std = torch.Tensor(self.offset_normalize_std).to(device)[None, None, :]
        roi_bbox_pred = roi_bbox_pred.view(roi_bbox_pred.shape[0], -1, 4)
        roi_bbox_pred = (roi_bbox_pred * std) + mean

        roi = torch.FloatTensor(roi).to(
            device).view(-1, 1, 4).expand_as(roi_bbox_pred)
        pred_bbox = offset2bbox(roi.cpu().numpy().reshape((-1, 4)),
                                roi_bbox_pred.cpu().numpy().reshape((-1, 4)))
        pred_bbox = torch.FloatTensor(pred_bbox).to(device)
        pred_bbox = pred_bbox.view(-1, self.n_class * 4)

        pred_bbox[:, 0::2] = (pred_bbox[:, 0::2]).clamp(min=0, max=img_size[0])
        pred_bbox[:, 1::2] = (pred_bbox[:, 1::2]).clamp(min=0, max=img_size[1])

        prob = F.softmax(roi_cls_scores, dim=1)

        bbox, label, score = self.suppress(pred_bbox, prob)

        self.train()
        return bbox, label, score

    def suppress(self, pred_bbox, prob):
        bbox = list()
        label = list()
        score = list()

        for i in range(1, self.n_class):
            pred_bbox_i = pred_bbox.view(-1, self.n_class, 4)[:, i, :]
            prob_i = prob[:, i]
            mask = (prob_i > self.score_thresh)
            pred_bbox_i = pred_bbox_i[mask, :]
            prob_i = prob_i[mask]
            index_keep = nms(pred_bbox_i, prob_i, self.nms_thresh)
            bbox.append(pred_bbox_i[index_keep].cpu().numpy())
            label.append((i - 1) * np.ones((len(index_keep),)))
            score.append(prob_i[index_keep].cpu().numpy())
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)

        return bbox, label, score

    def get_optimizer(self):
        self.optimizer = \
            torch.optim.SGD(self.parameters(), lr=config.lr,
                            momentum=0.9, weight_decay=config.weight_decay)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer
