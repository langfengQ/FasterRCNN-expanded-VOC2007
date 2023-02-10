import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def read_img(path, color=True):
    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=np.float32)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return np.expand_dims(img, 0)
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))


def visualize(img, bbox):
    plt.figure(str(img[0,0,0] + img[0,50,125]))
    bbox = bbox.astype(np.int)

    for i in range(bbox.shape[0]):
        b = bbox[i, :]
        img[:, b[0], b[1]:b[3]] = 255
        img[:, b[2], b[1]:b[3]] = 255
        img[:, b[0]:b[2], b[3]] = 255
        img[:, b[0]:b[2], b[1]] = 255

    plt.imshow(img.transpose((1, 2, 0)).astype(np.uint8))

    plt.show()




