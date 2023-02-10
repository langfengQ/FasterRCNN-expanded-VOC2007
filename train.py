from utils.config import config
from dataloader.dataset import Dataset
import torch
from train_tools.trainer import FatserRCNNTrainer
from model.faster_rcnn_vgg16 import FasterRCNN_VGG16, FasterRCNN_ResNet50, FasterRCNN_ResNet34
from tqdm import tqdm
import os
from utils.vis_tools import *
from utils.eval_tools import eval_detection_voc
from dataloader.my_transform import inverse_normalize
from tensorboardX import SummaryWriter


def eval(epoch, dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for i, (img, bbox, label, scale, difficult) in tqdm(enumerate(dataloader), total=len(dataloader), ncols=100,
                                                        desc='Epoch:{}'.format(epoch)):
        img, scale = img.to(config.device), scale.item()
        pred_bbox, pred_label, pred_score = faster_rcnn.predict(
            img, scale=1.0, eval=True)

        gt_bboxes += [bbox.numpy()[0]]
        gt_labels += [label.numpy()[0]]
        gt_difficults += [difficult.numpy()[0]]
        pred_bboxes += [pred_bbox]
        pred_labels += [pred_label]
        pred_scores += [pred_score]

        # if i == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def execute_one_epoch(epoch, trainer, train_loader, test_loader, tensorboard):
    with tqdm(enumerate(train_loader), total=len(train_loader), ncols=100, desc='Epoch:{}'.format(epoch)) as t:
        for i, (img, bbox, label, scale, _) in t:
            img, bbox, label, scale = img.to(config.device), bbox.to(
                config.device), label.to(config.device), scale.item()
            trainer.train_step(img, bbox, label, scale=1.0)

            if (i + 1) % config.plot_every == 0:

                # plot loss
                # trainer.vis.plot_many(trainer.get_meter_data())
                for caption, v in trainer.get_meter_data().items():
                    tensorboard.add_scalar(
                        caption, v, i + len(train_loader) * epoch)
                # t.set_postfix(trainer.get_meter_data())

                # plot groudtruth bboxes
                ori_img_ = inverse_normalize(img[0])
                gt_img = visdom_bbox(ori_img_,
                                     bbox[0].cpu().numpy(),
                                     label[0].cpu().numpy(),
                                     )
                trainer.vis.img('gt_img', gt_img)

                # plot predicted bboxes
                pred_bbox, pred_label, pred_score = trainer.faster_rcnn.predict(
                    img, scale=1.0, eval=False)
                pred_img = visdom_bbox(ori_img_,
                                       pred_bbox,
                                       pred_label,
                                       pred_score,
                                       )
                trainer.vis.img('pred_img', pred_img)

                trainer.vis.text(
                    str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                trainer.vis.img(
                    'roi_cm', torch.FloatTensor(trainer.roi_cm.conf))

    eval_result = eval(epoch, test_loader,
                       trainer.faster_rcnn, test_num=config.test_num)

    return eval_result


def train(**kwargs):
    config._parse(kwargs)

    train_dataset = Dataset(config, split='trainval', use_difficult=False)
    test_dataset = Dataset(config, split='test', use_difficult=True)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=1,
                                               shuffle=True,
                                               # pin_memory=True,
                                               num_workers=config.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              num_workers=config.test_num_workers,
                                              shuffle=False,
                                              pin_memory=True
                                              )
    faster_rcnn = FasterRCNN_VGG16()
    # faster_rcnn = FasterRCNN_ResNet50()
    # faster_rcnn = FasterRCNN_ResNet34()
    trainer = FatserRCNNTrainer(faster_rcnn).to(config.device)

    tensorboard = SummaryWriter(config.tensorboard_path)

    trainer.vis.text(train_dataset.voc.label_names, win='labels')

    best_map = 0

    for epoch in range(config.epoch):
        trainer.reset_meters()

        eval_result = execute_one_epoch(
            epoch, trainer, train_loader, test_loader, tensorboard)

        # trainer.vis.plot('test_map', eval_result['map'])
        tensorboard.add_scalar('test_map / epoch', eval_result['map'], epoch)
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'epoch:{}, lr:{}, map:{}, loss:{}'.format(str(epoch),
                                                             str(lr_),
                                                             str(eval_result['map']),
                                                             str(trainer.get_meter_data()))
        trainer.vis.log(log_info)

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(
                best_map=best_map, save_optimizer=False, save_path=config.checkpoints_path)
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(config.lr_decay)
        if epoch == 14:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(config.lr_decay)


if __name__ == '__main__':
    train(env='fasterrcnn', plot_every=500, epoch=19)
