import torch


def fast_rcnn_reg_loss(pre_offset, gt_offset, gt_label, sigma):

    in_weight = (gt_label > 0).float().view(-1, 1).expand_as(gt_offset)
    loss = smooth_l1_loss(pre_offset, gt_offset, in_weight, sigma)
    loss /= (torch.sum(gt_label >= 0).float())

    return loss


def smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()