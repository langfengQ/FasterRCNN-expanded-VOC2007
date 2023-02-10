from model.bbox_tools import *


class ProposalTargetCreator:
    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25,
                 pos_iou_thresh=0.5,
                 neg_iou_thresh_high=0.5,
                 neg_iou_thresh_low=0.0,
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_high = neg_iou_thresh_high
        self.neg_iou_thresh_low = neg_iou_thresh_low  # NOTE:default 0.1 in py-faster-rcnn

    def __call__(self, roi, bbox, label, offset_normalize_mean, offset_normalize_std):
        n_bbox = bbox.shape[0]
        roi = np.concatenate((roi, bbox), axis=0)

        pos_roi_num = np.round(self.n_sample * self.pos_ratio)

        iou = bbox_iou(roi, bbox)

        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)
        gt_roi_label = label[gt_assignment] + 1  # zero for background

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        pos_roi_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_num = int(min(pos_roi_num, pos_roi_index.shape[0]))
        if pos_roi_num > 0:
            pos_roi_index = np.random.choice(pos_roi_index, pos_roi_num, replace=False)

        # Select background RoIs as those within [neg_iou_thresh_lo, neg_iou_thresh_hi)
        neg_roi_index = np.where((max_iou >= self.neg_iou_thresh_low) & (max_iou < self.neg_iou_thresh_high))[0]
        neg_roi_num = self.n_sample - pos_roi_num
        neg_roi_num = int(min(neg_roi_num, neg_roi_index.shape[0]))
        if neg_roi_num > 0:
            neg_roi_index = np.random.choice(neg_roi_index, neg_roi_num, replace=False)

        keep_index = np.append(pos_roi_index, neg_roi_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_num:] = 0
        sample_roi = roi[keep_index, :]

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_offset = bbox2offset(sample_roi, bbox[gt_assignment[keep_index], :])
        gt_roi_offset = (gt_roi_offset - np.array(offset_normalize_mean)[None, :]) \
                        / np.array(offset_normalize_std)[None, :]

        return sample_roi, gt_roi_label, gt_roi_offset


class AnchorTargetCreator:
    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7,
                 neg_iou_thresh=0.3,
                 pos_ratio=0.5,
                 ):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, anchor, bbox, img_size):
        n_anchor = anchor.shape[0]

        inside_index = np.where((anchor[:, 0] >= 0) &
                                (anchor[:, 1] >= 0) &
                                (anchor[:, 2] <= img_size[0]) &
                                (anchor[:, 3] <= img_size[1])
                                )[0]

        anchor = anchor[inside_index, :]
        argmax_ious, label = self.create_label(anchor, bbox)
        offset = bbox2offset(anchor, bbox[argmax_ious, :])

        label_fill = np.ones((n_anchor,), dtype=label.dtype) * -1
        label_fill[inside_index] = label

        offset_fill = np.zeros((n_anchor,) + offset.shape[1:], dtype=offset.dtype)
        offset_fill[inside_index, :] = offset

        return offset_fill, label_fill

    def create_label(self, anchor, bbox):
        label = np.ones((anchor.shape[0],), dtype=np.int32) * -1

        ious = bbox_iou(anchor, bbox)
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious.max(axis=1)
        gt_max_ious = ious.max(axis=0)
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        # take 256*0.5=128 positive for rpn training
        label[gt_argmax_ious] = 1
        label[max_ious >= self.pos_iou_thresh] = 1

        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # take 256*0.5=128 negative for rpn training
        label[max_ious < self.neg_iou_thresh] = 0

        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label



