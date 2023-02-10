import os


class Config:
    # data
    # path = '/home/fenglang/project/cv_homework/faster_rcnn/code'
    path = os.getcwd()
    voc_data_dir = path + '/My_Dataset/VOC2007/'
    min_size = 600  # image resize
    max_size = 1000  # image resize
    num_workers = 0
    test_num_workers = 0

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # visualization
    env = 'fasterrcnn'  # visdom env
    port = 8097
    plot_every = 40  # vis every N iter

    # preset
    data = 'voc'
    pretrained_model = 'vgg16'
    # pretrained_model = 'resnet34'
    device = 'cuda:0'

    # training
    epoch = 14

    # checkpoints
    checkpoints_path = path + '/checkpoints/best_fasterrcnn_vocwithcoco_run2'

    # tensorboard
    tensorboard_path = path + '/summaries/debug/'

    use_adam = False  # Use Adam optimizer
    use_chainer = False  # try match everything as chainer
    use_drop = False  # use dropout in RoIHead
    # debug
    debug_file = path + '/tmp/debugf'

    test_num = 10000
    # model
    load_path = None

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() if k[0] != '_'}


config = Config()
