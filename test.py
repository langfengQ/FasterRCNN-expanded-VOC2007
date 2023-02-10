from utils.config import config
from dataloader.dataset import Dataset
import torch
from model.faster_rcnn_vgg16 import FasterRCNN_VGG16, FasterRCNN_ResNet34
from tqdm import tqdm
import os
from utils.vis_tools import *
from utils.eval_tools import eval_detection_voc
from dataloader.my_transform import inverse_normalize


def eval(dataloader, faster_rcnn, vis, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for i, (img, bbox, label, scale, difficult) in tqdm(enumerate(dataloader), total=len(dataloader), ncols=100,):
        img, scale = img.to(config.device), scale.item()
        pred_bbox, pred_label, pred_score = faster_rcnn.predict(img, scale=1.0, eval=True)

        gt_bboxes += [bbox.numpy()[0]]
        gt_labels += [label.numpy()[0]]
        gt_difficults += [difficult.numpy()[0]]
        pred_bboxes += [pred_bbox]
        pred_labels += [pred_label]
        pred_scores += [pred_score]

        # if i == test_num: break

        if (i + 1) % config.plot_every == 0:

            # plot loss
            # trainer.vis.plot_many(trainer.get_meter_data())
            # t.set_postfix(trainer.get_meter_data())
                
            # plot groudtruth bboxes

            ori_img_ = inverse_normalize(img[0])
            gt_img = visdom_bbox(ori_img_,
                                bbox[0].cpu().numpy(),
                                label[0].cpu().numpy(),
                                )
            vis.img('gt_img', gt_img)

            # plot predicted bboxes
            pred_img = visdom_bbox(ori_img_,
                                pred_bbox,
                                pred_label,
                                pred_score,
                                )
            vis.img('pred_img', pred_img)

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def test(**kwargs):
    config._parse(kwargs)

    test_dataset = Dataset(config, split='test', use_difficult=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                       batch_size=1,
                                       num_workers=config.test_num_workers,
                                       shuffle=True,
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNN_VGG16().to(config.device)

    vis = Visualizer(env=config.env)

    state_dict = torch.load(config.path + '/checkpoints/best_fasterrcnn_vocwithcoco_run2')
    if 'model' in state_dict:
        faster_rcnn.load_state_dict(state_dict['model'])
    else:  # legacy way, for backward compatibility
        faster_rcnn.load_state_dict(state_dict)


    eval_result = eval(test_loader, faster_rcnn, vis, test_num=config.test_num)

    print('map: ', eval_result)


if __name__ == '__main__':
    test(env='fasterrcnn_test', plot_every=100)
    