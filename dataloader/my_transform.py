from skimage import transform
from torchvision import transforms
from utils.config import config
from utils.utils import visualize
import torch
import numpy as np


class Normailize():
    def __init__(self):
        pass

    def __call__(self, img, **kwargs):
        img = img / 255.
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        img = normalize(torch.from_numpy(img))

        return {'img': img, **kwargs}


class Resize():
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img, bbox, **kwargs):
        # img
        C, H, W = img.shape
        scale1 = self.min_size / min(H, W)
        scale2 = self.max_size / max(H, W)
        scale = min(scale1, scale2)
        img = transform.resize(img, (C, H * scale, W * scale), mode='reflect', anti_aliasing=False)

        # bbox
        _, o_H, o_W = img.shape
        bbox = bbox.copy()
        y_scale = float(o_H) / H
        x_scale = float(o_W) / W
        bbox[:, 0] = y_scale * bbox[:, 0]
        bbox[:, 2] = y_scale * bbox[:, 2]
        bbox[:, 1] = x_scale * bbox[:, 1]
        bbox[:, 3] = x_scale * bbox[:, 3]

        return {'img': img, 'bbox': bbox, 'scale': scale, **kwargs}


class Random_flip():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bbox, **kwargs):
        if np.random.random() < self.p:
            # img
            img = img[:, :, ::-1]

            # bbox
            _, H, W = img.shape
            x_max = W - bbox[:, 1]
            x_min = W - bbox[:, 3]
            bbox[:, 1] = x_min
            bbox[:, 3] = x_max

        return {'img': img, 'bbox': bbox, **kwargs}


def inverse_normalize(img):

    mean = np.array([0.485, 0.456, 0.406])[:, np.newaxis, np.newaxis]
    std = np.array([0.229, 0.224, 0.225])[:, np.newaxis, np.newaxis]
    if len(img.shape) == 4:
        mean = np.array([0.485, 0.456, 0.406])[np.newaxis, :, np.newaxis, np.newaxis]
        std = np.array([0.229, 0.224, 0.225])[np.newaxis, :, np.newaxis, np.newaxis]
    ori_img = (img.cpu().numpy() * std + mean).clip(min=0, max=1) * 255

    return ori_img