from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab, os, cv2, shutil
from lxml import etree, objectify
from tqdm import tqdm
import random
from PIL import Image


pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir = './coco_data'
CK5cats = ['boat', 'bottle', 'chair']
CK5cats_voc = [ 'airplane',
                'bicycle',
                'bird',
                'boat',
                'bottle',
                'bus',
                'car',
                'cat',
                'chair',
                'cow',
                'dining table',
                'dog',
                'horse',
                'motorcycle',
                'person',
                'pizza',
                'sheep',
                'couch',
                'train',
                'tv',
]

VOC_BBOX_LABEL_NAMES = [
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
    'tvmonitor']

CKdir = "./result_data"
CKimg_dir = CKdir + "/" + "images"
CKanno_dir = CKdir + "/" + "Annotations"


def mkr(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def showimg(coco, dataType, img, CK5Ids):
    global dataDir
    I = io.imread('%s/%s/%s' % (dataDir, dataType, img['file_name']))
    plt.imshow(I)
    plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=CK5Ids, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    plt.show()


def save_annotations(dataType, filename, objs):
    annopath = CKanno_dir + "/coco" + filename[:-3] + "xml"
    img_path = dataDir + "/" + dataType + "/" + filename
    dst_path = CKimg_dir + "/coco" + filename
    img = cv2.imread(img_path)
    im = Image.open(img_path)
    if im.mode != "RGB":
        print(filename + " not a RGB image")
        im.close()
        return
    im.close()
    shutil.copy(img_path, dst_path)
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('1'),
        E.filename(filename),
        E.source(
            E.database('CKdemo'),
            E.annotation('VOC'),
            E.image('CK')
        ),
        E.size(
            E.width(img.shape[1]),
            E.height(img.shape[0]),
            E.depth(img.shape[2])
        ),
        E.segmented(0)
    )
    for obj in objs:
        E2 = objectify.ElementMaker(annotate=False)
        anno_tree2 = E2.object(
            E.name(obj[0]),
            E.pose(),
            E.truncated("0"),
            E.difficult(0),
            E.bndbox(
                E.xmin(obj[2]),
                E.ymin(obj[3]),
                E.xmax(obj[4]),
                E.ymax(obj[5])
            )
        )
        anno_tree.append(anno_tree2)
    etree.ElementTree(anno_tree).write(annopath, pretty_print=True)


def showbycv(coco, dataType, img, classes, CK5Ids, l):
    global dataDir
    filename = img['file_name']
    filepath = '%s/%s/%s' % (dataDir, dataType, filename)
    I = cv2.imread(filepath)
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=CK5Ids, iscrowd=None)
    anns = coco.loadAnns(annIds)
    objs = []
    save_flag = True
    if len(annIds) <= 8 and random.random() < 1700./l:
        for ann in anns:
            name = classes[ann['category_id']]
            if name == 'pizza':
                save_flag = False
                break
            if name in CK5cats_voc:
                name = VOC_BBOX_LABEL_NAMES[CK5cats_voc.index(name)]
                if 'bbox' in ann:
                    bbox = ann['bbox']
                    xmin = (int)(bbox[0])
                    ymin = (int)(bbox[1])
                    xmax = (int)(bbox[2] + bbox[0])
                    ymax = (int)(bbox[3] + bbox[1])
                    obj = [name, 1.0, xmin, ymin, xmax, ymax]
                    objs.append(obj)
                    cv2.rectangle(I, (xmin, ymin), (xmax, ymax), (255, 0, 0))
                    cv2.putText(I, name, (xmin, ymin), 3, 0.5, (0, 0, 255))
        # visualize(I)
        if save_flag:
            # print('save')
            save_annotations(dataType, filename, objs)


def catid2name(coco):
    classes = dict()
    for cat in coco.dataset['categories']:
        classes[cat['id']] = cat['name']
        # print(str(cat['id'])+":"+cat['name'])
    return classes


def get_CK5():
    mkr(CKimg_dir)
    mkr(CKanno_dir)
    dataTypes = ['train2017']
    for dataType in dataTypes:
        annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
        coco = COCO(annFile)
        CK5Ids = coco.getCatIds(catNms=CK5cats_voc)
        classes = catid2name(coco)
        for srccat in CK5cats:
            print(dataType + ":" + srccat)
            catIds = coco.getCatIds(catNms=[srccat])
            imgIds = coco.getImgIds(catIds=catIds)
            # imgIds=imgIds[0:100]
            for imgId in tqdm(imgIds):
                img = coco.loadImgs(imgId)[0]
                showbycv(coco, dataType, img, classes, CK5Ids, len(imgIds))
                # showimg(coco,dataType,img,CK5Ids)


# split train and test for training
def split_traintest(trainratio=1, testratio=0):
    dataset_dir = CKdir
    files = os.listdir(CKimg_dir)
    trainvals = []
    tests = []
    random.shuffle(files)
    for i in range(len(files)):
        filepath = CKimg_dir + "/" + files[i][:-3] + "jpg"
        if i <= trainratio * len(files):
            trainvals.append(files[i])
        elif i > trainratio * len(files) and i < (trainratio + testratio) * len(files):
            tests.append(files[i])
    # write txt files for yolo
    with open(dataset_dir + "/trainval.txt", "w")as f:
        for line in trainvals:
            line = CKimg_dir + "/" + line
            f.write(line + "\n")
    with open(dataset_dir + "/test.txt", "w") as f:
        for line in tests:
            line = CKimg_dir + "/" + line
            f.write(line + "\n")
    # write files for voc
    maindir = dataset_dir + "/" + "ImageSets/Main"
    mkr(maindir)
    with open(maindir + "/trainval.txt", "w") as f:
        for line in trainvals:
            line = line[:line.rfind(".")]
            f.write(line + "\n")
    with open(maindir + "/test.txt", "w") as f:
        for line in tests:
            line = line[:line.rfind(".")]
            f.write(line + "\n")
    print("spliting done")



def visualize(img):
    import numpy as np
    img = np.transpose(img, (2, 0, 1))
    # bbox = np.stack(bbox, axis=0)
    plt.figure(str(img[0,0,0] + img[0,50,125]))
    # bbox = bbox.astype(np.int)

    # for i in range(bbox.shape[0]):
    #     b = bbox[i, :]
    #     img[:, b[0], b[1]:b[3]] = 255
    #     img[:, b[2], b[1]:b[3]] = 255
    #     img[:, b[0]:b[2], b[3]] = 255
    #     img[:, b[0]:b[2], b[1]] = 255

    plt.imshow(img.transpose((1, 2, 0)).astype(np.uint8))

    plt.show()

if __name__ == "__main__":
    # get_CK5()
    split_traintest()
