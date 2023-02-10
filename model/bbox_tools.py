import six
import numpy as np


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], scales=[8, 16, 32]):
    center_h, center_w = base_size / 2., base_size / 2.

    anchor_base = np.zeros([len(ratios)*len(scales), 4], dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(scales)):
            h = base_size * scales[j] * np.sqrt(ratios[i])
            w = base_size * scales[j] / np.sqrt(ratios[i])

            index = len(scales) * i + j
            anchor_base[index, 0] = center_h - h / 2.
            anchor_base[index, 1] = center_w - w / 2.
            anchor_base[index, 2] = center_h + h / 2.
            anchor_base[index, 3] = center_w + w / 2.

    return anchor_base


def enumerate_shifted_anchor(anchor_base, feature_size, feat_stride):
    height, width = feature_size[0], feature_size[1]
    shift_h = np.arange(0, height * feat_stride, feat_stride)
    shift_w = np.arange(0, width * feat_stride, feat_stride)

    shift_w, shift_h = np.meshgrid(shift_w, shift_h)

    shift = np.stack((shift_h.ravel(), shift_w.ravel(), shift_h.ravel(), shift_w.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)

    return anchor


def offset2bbox(anchors, rpn_bbox_offset):

    # if anchors.shape[0] == 0:
    #     return np.zeros((0, 4), dtype=rpn_bbox_offset.dtype)

    # anchors = anchors.astype(anchors.dtype, copy=False)

    height = anchors[:, 2] - anchors[:, 0]
    width = anchors[:, 3] - anchors[:, 1]
    center_h = anchors[:, 0] + 0.5 * height
    center_w = anchors[:, 1] + 0.5 * width

    Ay = rpn_bbox_offset[:, 0]
    Ax = rpn_bbox_offset[:, 1]
    Ah = rpn_bbox_offset[:, 2]
    Aw = rpn_bbox_offset[:, 3]

    center_h_ = height * Ay + center_h
    center_w_ = width * Ax + center_w
    height_ = height * np.exp(Ah)
    width_ = width * np.exp(Aw)

    pre_bbox = np.zeros(rpn_bbox_offset.shape, dtype=rpn_bbox_offset.dtype)
    pre_bbox[:, 0] = center_h_ - 0.5 * height_
    pre_bbox[:, 1] = center_w_ - 0.5 * width_
    pre_bbox[:, 2] = center_h_ + 0.5 * height_
    pre_bbox[:, 3] = center_w_ + 0.5 * width_

    return pre_bbox


def bbox2offset(roi, bbox):
    roi_height = roi[:, 2] - roi[:, 0]
    roi_width = roi[:, 3] - roi[:, 1]
    roi_y = roi[:, 0] + 0.5 * roi_height
    roi_x = roi[:, 1] + 0.5 * roi_width

    bbox_height = bbox[:, 2] - bbox[:, 0]
    bbox_width = bbox[:, 3] - bbox[:, 1]
    bbox_y = bbox[:, 0] + 0.5 * bbox_height
    bbox_x = bbox[:, 1] + 0.5 * bbox_width

    eps = np.finfo(roi_height.dtype).eps
    roi_height = np.maximum(roi_height, eps)
    roi_width = np.maximum(roi_width, eps)

    Ay = (bbox_y - roi_y) / roi_height
    Ax = (bbox_x - roi_x) / roi_width
    Ah = np.log(bbox_height / roi_height)
    Aw = np.log(bbox_width / roi_width)

    # gt_bbox_offset = np.vstack((Ay, Ax, Ah, Aw)).transpose()
    gt_roi_offset = np.concatenate((Ay[:, np.newaxis], Ax[:, np.newaxis],
                                    Ah[:, np.newaxis], Aw[:, np.newaxis]), axis=1)
    return gt_roi_offset


def bbox_iou(roi, bbox):
    top_left = np.maximum(roi[:, np.newaxis, :2], bbox[np.newaxis, :, :2])
    bottom_right = np.minimum(roi[:, np.newaxis, 2:], bbox[np.newaxis, :, 2:])

    area_i = np.prod(bottom_right - top_left, axis=2) * (top_left < bottom_right).all(axis=2)
    area_a = np.prod(roi[:, 2:] - roi[:, :2], axis=1)
    area_b = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)
    area_u = area_a[:, np.newaxis] + area_b[np.newaxis, :] - area_i

    return area_i / area_u

