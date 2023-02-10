import os
import xml.etree.ElementTree as ET
import numpy as np
from utils.utils import *
from .my_transform import *
from torchvision.transforms import Compose

VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pizza',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')


class VOCBboxDataset():
    def __init__(self, data_dir, split='trainval', use_difficult=False, return_difficult=False,):

        id_list_file = os.path.join(data_dir, 'ImageSets/Main_partialcoco/{}.txt'.format(split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        id_ = self.ids[item]
        anno = ET.parse(os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox, label, difficult = [], [], []

        for obj in anno.findall('object'):
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        difficult = np.array(difficult).astype(np.uint8)

        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_img(img_file, color=True)

        return img, bbox, label, difficult


class Dataset():
    def __init__(self, config, split='trainval', use_difficult=False):
        self.config = config
        self.voc = VOCBboxDataset(config.voc_data_dir, split=split, use_difficult=use_difficult)
        self.transform = Transform(split)

    def __getitem__(self, item):
        img, bbox, label, difficult = self.voc[item]
        img, bbox, scale = self.transform(img=img, bbox=bbox)

        return img, bbox, label, scale, difficult

    def __len__(self):
        return self.voc.__len__()


class Transform(object):

    def __init__(self, split):
        if split == 'trainval':
            self.trans = [Resize(),
                          Random_flip(p=0.5),
                          Normailize()]
        else:
            self.trans = [Resize(),
                          Normailize()]

    def __call__(self, **kwargs):
        for t in self.trans:
            kwargs = t(**kwargs)

        return kwargs['img'], kwargs['bbox'], kwargs['scale']