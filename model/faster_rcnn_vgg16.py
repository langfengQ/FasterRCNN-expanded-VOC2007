import torch.nn as nn
from .faster_rcnn import FasterRCNN
from utils.config import config
from model.bbox_tools import *
from torchvision.models import vgg16, resnet50, resnet34
import torch.nn.functional as F
from torchvision.ops import nms
import torch
from torchvision.ops import RoIPool


class FasterRCNN_VGG16(FasterRCNN):
    @property
    def feat_stride(self):
        return 16

    def __init__(self,
                 n_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32],
                 ):
        model = vgg16(not config.load_path)

        features = list(model.features)[:30]
        classifier = list(model.classifier)

        del classifier[6]
        if not config.use_drop:
            del classifier[5]
            del classifier[2]
        classifier = nn.Sequential(*classifier)

        for layer in features[:10]:
            for p in layer.parameters():
                p.requires_grad = False

        extractor = nn.Sequential(*features)
        rpn = RegionProposalNetwork(in_channels=512,
                                    mid_channels=512,
                                    ratios=ratios,
                                    anchor_scales=anchor_scales,
                                    feat_stride=self.feat_stride,
                                    )

        head = RoIHead(n_class=n_class + 1,
                       roi_size=7,
                       spatial_scale=(1. / self.feat_stride),
                       classifier=classifier
                       )

        super(FasterRCNN_VGG16, self).__init__(extractor, rpn, head)


class FasterRCNN_ResNet50(FasterRCNN):
    @property
    def feat_stride(self):
        return 16

    def __init__(self,
                 n_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32],
                 ):

        model = resnet50(not config.load_path)

        features = []
        features += [model.conv1, model.bn1, model.relu, model.maxpool]
        features += [model.layer1, model.layer2, model.layer3]

        for layer in features[:5]:
            for p in layer.parameters():
                p.requires_grad = False

        extractor = nn.Sequential(*features)
        classifier = model.layer4
        rpn = RegionProposalNetwork(in_channels=1024,
                                    mid_channels=512,
                                    ratios=ratios,
                                    anchor_scales=anchor_scales,
                                    feat_stride=self.feat_stride,
                                    )

        head = RoIHead(n_class=n_class + 1,
                       roi_size=7,
                       spatial_scale=(1. / self.feat_stride),
                       classifier=classifier,
                       channel=2048,
                       resnet=True)

        super(FasterRCNN_ResNet50, self).__init__(extractor, rpn, head)


class FasterRCNN_ResNet34(FasterRCNN):
    @property
    def feat_stride(self):
        return 16

    def __init__(self,
                 n_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32],
                 ):

        model = resnet34(not config.load_path)

        features = []
        features += [model.conv1, model.bn1, model.relu, model.maxpool]
        features += [model.layer1, model.layer2, model.layer3]

        for layer in features[:5]:
            for p in layer.parameters():
                p.requires_grad = False

        extractor = nn.Sequential(*features)
        classifier = model.layer4
        rpn = RegionProposalNetwork(in_channels=256,
                                    mid_channels=512,
                                    ratios=ratios,
                                    anchor_scales=anchor_scales,
                                    feat_stride=self.feat_stride,
                                    )

        head = RoIHead(n_class=n_class + 1,
                       roi_size=7,
                       spatial_scale=(1. / self.feat_stride),
                       classifier=classifier,
                       channel=512,
                       resnet=True)

        super(FasterRCNN_ResNet34, self).__init__(extractor, rpn, head)


class RegionProposalNetwork(nn.Module):
    def __init__(self,
                 in_channels=512,
                 mid_channels=512,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32],
                 feat_stride=16,
                 proposal_creator_params={},
                 ):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base(
            scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride
        n_anchor = self.anchor_base.shape[0]
        # self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        # self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   )
        self.rpn_cls_score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.rpn_bbox_pred = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

        kaiming_uniform_init(self.conv1)
        normal_init(self.rpn_cls_score)
        normal_init(self.rpn_bbox_pred)

        self.proposal_layer = ProposalLayer(**proposal_creator_params)

    def forward(self, x, img_size, scale):
        self.bs, _, hh, ww = x.shape
        self.anchors = enumerate_shifted_anchor(self.anchor_base,
                                                feat_stride=self.feat_stride,
                                                feature_size=(hh, ww),
                                                )

        n_anchor = self.anchors.shape[0] // (hh * ww)

        x = self.conv1(x)

        # rpn_cls_score
        rpn_cls_scores = self.rpn_cls_score(x)

        rpn_cls_scores = rpn_cls_scores.permute(0, 2, 3, 1).contiguous()
        rpn_softmax_scores = F.softmax(
            rpn_cls_scores.view(self.bs, hh, ww, n_anchor, 2), dim=4)
        rpn_softmax_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous().view(
            self.bs, -1)

        rpn_cls_scores = rpn_cls_scores.view(self.bs, -1, 2)

        # rpn_bbox_pred
        rpn_offsets = self.rpn_bbox_pred(x)
        rpn_offsets = rpn_offsets.permute(
            0, 2, 3, 1).contiguous().view(self.bs, -1, 4)

        rois, rois_index = self.proposal(
            rpn_offsets, rpn_softmax_scores, img_size, scale)

        return rois, rois_index, rpn_offsets, rpn_cls_scores, self.anchors

    def proposal(self, rpn_bbox_offset, rpn_softmax_scores, img_size, scale):
        rois, rois_index = [], []
        for i in range(self.bs):
            roi = self.proposal_layer(self.anchors,
                                      rpn_bbox_offset[i],
                                      rpn_softmax_scores[i],
                                      img_size,
                                      scale,
                                      )
            rois.append(roi)
            batch_index = i * np.ones((roi.shape[0],), dtype=np.int32)
            rois_index.append(batch_index)

        rois = np.concatenate(rois, axis=0)
        rois_index = np.concatenate(rois_index, axis=0)

        return rois, rois_index


class ProposalLayer(nn.Module):
    def __init__(self,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16
                 ):
        super(ProposalLayer, self).__init__()
        self.nms_thresh = nms_thresh
        self.min_size = min_size
        if self.training:
            self.n_pre_nms = n_train_pre_nms
            self.n_post_nms = n_train_post_nms
        else:
            self.n_pre_nms = n_test_pre_nms
            self.n_post_nms = n_test_post_nms

    def forward(self, anchors, rpn_bbox_offset, rpn_softmax_scores, img_size, scale):
        device = rpn_bbox_offset.device
        rpn_bbox_offset = rpn_bbox_offset.cpu().data.numpy()
        rpn_softmax_scores = rpn_softmax_scores.cpu().data.numpy()
        roi = offset2bbox(anchors, rpn_bbox_offset)

        # clip
        roi[:, 0] = np.clip(roi[:, 0], 0, img_size[0])
        roi[:, 2] = np.clip(roi[:, 2], 0, img_size[0])
        roi[:, 1] = np.clip(roi[:, 1], 0, img_size[1])
        roi[:, 3] = np.clip(roi[:, 3], 0, img_size[1])

        # remove small rois
        min_size = self.min_size * scale
        roi_h = roi[:, 2] - roi[:, 0]
        roi_w = roi[:, 3] - roi[:, 1]
        index_keep = np.where((roi_h >= min_size) & (roi_w >= min_size))[0]

        roi = roi[index_keep, :]
        rpn_softmax_scores = rpn_softmax_scores[index_keep]

        # take top pre_nms_topN (e.g. 6000).
        order = rpn_softmax_scores.ravel().argsort()[::-1]
        index_keep = order[:self.n_pre_nms]
        roi = roi[index_keep, :]
        rpn_softmax_scores = rpn_softmax_scores[index_keep]

        # Take after_nms_topN (e.g. 300).
        index_keep = nms(torch.from_numpy(roi).to(device),
                         torch.from_numpy(rpn_softmax_scores).to(device),
                         self.nms_thresh,
                         )
        index_keep = index_keep[:self.n_post_nms].cpu().numpy()
        roi = roi[index_keep, :]

        return roi


class RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier, channel=4096, resnet=False):
        super(RoIHead, self).__init__()
        self.resnet = resnet
        self.classifier = classifier
        self.bbox_pred = nn.Linear(channel, n_class * 4)
        self.cls_score = nn.Linear(channel, n_class)
        kaiming_normal_init(self.bbox_pred)
        kaiming_normal_init(self.cls_score)

        self.n_class = n_class
        self.roipool = RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, x, rois, rois_index):
        rois = torch.FloatTensor(rois).to(x.device)
        rois_index = torch.FloatTensor(rois_index).to(x.device)
        index_and_rois = torch.cat([rois_index[:, None], rois], dim=1)

        index_and_rois = index_and_rois[:, [0, 2, 1, 4, 3]].contiguous()

        x = self.roipool(x, index_and_rois)
        if self.resnet:
            x = self.classifier(x)
            x = x.mean(3).mean(2)
            x = x.view(x.size()[0], -1)
        else:
            x = x.view(x.shape[0], -1)
            x = self.classifier(x)
        roi_bbox_pred = self.bbox_pred(x)
        roi_cls_scores = self.cls_score(x)

        return roi_bbox_pred, roi_cls_scores


# def normal_init(modules, mean, stddev):
#     if isinstance(modules, nn.Sequential):
#         for m in modules:
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#                 m.weight.data.normal_(mean, stddev)
#                 m.bias.data.zero_()
#     else:
#         modules.weight.data.normal_(mean, stddev)
#         modules.bias.data.zero_()

def kaiming_uniform_init(modules):
    if isinstance(modules, nn.Sequential):
        for m in modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
    else:
        if isinstance(modules, nn.Conv2d) or isinstance(modules, nn.Linear):
            nn.init.kaiming_uniform_(modules.weight, a=1)
            nn.init.constant_(modules.bias, 0)

def normal_init(modules):
    if isinstance(modules, nn.Sequential):
        for m in modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)
    else:
        if isinstance(modules, nn.Conv2d) or isinstance(modules, nn.Linear):
            nn.init.normal_(modules.weight, std=0.01)
            nn.init.constant_(modules.bias, 0)

def kaiming_normal_init(modules):
    if isinstance(modules, nn.Sequential):
        for m in modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    else:
        if isinstance(modules, nn.Conv2d) or isinstance(modules, nn.Linear):
            nn.init.kaiming_normal_(modules.weight)
            nn.init.constant_(modules.bias, 0)