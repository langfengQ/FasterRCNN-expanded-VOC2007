import torch.nn as nn
import torch
from train_tools.creator_tools import *
from train_tools.loss_tools import *
from utils.vis_tools import Visualizer
from torchnet.meter import ConfusionMeter, AverageValueMeter
from collections import namedtuple
from utils.config import config
import torch.nn.functional as F
import time
import os


LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])

class FatserRCNNTrainer(nn.Module):
    def __init__(self, faster_rcnn):
        super(FatserRCNNTrainer, self).__init__()
        self.rpn_sigma = config.rpn_sigma
        self.roi_sigma = config.roi_sigma

        self.faster_rcnn = faster_rcnn
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()
        self.offset_normalize_mean = faster_rcnn.offset_normalize_mean
        self.offset_normalize_std = faster_rcnn.offset_normalize_std

        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

        self.vis = Visualizer(env=config.env)

        self.optimizer = faster_rcnn.get_optimizer()

    def get_losses(self, img, bbox, label, scale):
        device = img.device

        if bbox.shape[0] != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = img.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(img)

        # generate rois
        rois, rois_index, rpn_offsets, rpn_cls_scores, anchors \
            = self.faster_rcnn.rpn(features, img_size, scale)

        bbox = bbox[0].detach().cpu().numpy()
        label = label[0].detach().cpu().numpy()
        rpn_offset = rpn_offsets[0]
        rpn_cls_score = rpn_cls_scores[0]
        roi = rois

        # sample 128 rois for training
        sample_roi, gt_roi_label, gt_roi_offset \
            = self.proposal_target_creator(roi, bbox, label,
                                           self.offset_normalize_mean,
                                           self.offset_normalize_std,)

        sample_roi_index = np.zeros(sample_roi.shape[0])
        roi_bbox_offset, roi_cls_scores \
            = self.faster_rcnn.head(features,
                                    sample_roi,
                                    sample_roi_index)


        # ------------------ RPN losses -------------------#
        gt_rpn_offset, gt_rpn_label = self.anchor_target_creator(anchors, bbox, img_size)
        gt_rpn_label = torch.LongTensor(gt_rpn_label).to(device)
        gt_rpn_offset = torch.FloatTensor(gt_rpn_offset).to(device)
        rpn_reg_loss = fast_rcnn_reg_loss(
            rpn_offset,
            gt_rpn_offset,
            gt_rpn_label.data,
            self.rpn_sigma)

        rpn_cls_loss = F.cross_entropy(rpn_cls_score, gt_rpn_label, ignore_index=-1)
        
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = rpn_cls_score.detach().cpu().numpy()[gt_rpn_label.detach().cpu().numpy() > -1]
        self.rpn_cm.add(torch.FloatTensor(_rpn_score), _gt_rpn_label.data.long())

        # ------------------ ROI losses -------------------#
        n_sample = roi_bbox_offset.shape[0]
        roi_bbox_offset = roi_bbox_offset.view(n_sample, -1, 4)
        roi_offset = roi_bbox_offset[torch.arange(0, n_sample).long(), \
                              torch.LongTensor(gt_roi_label)]
        gt_roi_label = torch.LongTensor(gt_roi_label).to(device)
        gt_roi_offset = torch.FloatTensor(gt_roi_offset).to(device)
        roi_reg_loss = fast_rcnn_reg_loss(
            roi_offset,
            gt_roi_offset,
            gt_roi_label.data,
            self.roi_sigma)

        roi_cls_loss = F.cross_entropy(roi_cls_scores, gt_roi_label)

        self.roi_cm.add(roi_cls_scores.detach().cpu(), gt_roi_label.data.long())
        losses = [rpn_reg_loss, rpn_cls_loss, roi_reg_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)

    def train_step(self, img, bbox, label, scale):
        self.optimizer.zero_grad()
        losses = self.get_losses(img, bbox, label, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses

    def update_meters(self, losses):
        loss_dic = losses._asdict()
        for key, meter in self.meters.items():
            meter.add(loss_dic[key].item())

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}

    def save(self, save_optimizer=True, save_path=None, **kwargs):
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = config._state_dict()
        save_dict['other_info'] = kwargs
        save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = config.path + '/checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(save_dict, save_path)
        self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_config=False, ):
        state_dict = torch.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_config:
            config._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

