import os
import tensorflow as tf
from data.tfrecords import bytes_feature, int64_feature

from data.util import padding, open_image, load_disparity, load_pfm


class BaseExample():
    left = None
    right = None
    disparity = None
    path = None
    width = None
    height = None
    channels = 3
    name = None

    def __init__(self, path):
        for key in ['left', 'right', 'disparity']:
            if key not in path:
                raise RuntimeError('Key {} not in path'.format(key))
        self.name = os.path.split(path['left'])[-1].split('.')[0]
        self.size = (self.height, self.width, self.channels)

    def __str__(self):
        return self.name

    def to_tfrecords(self):
        return tf.train.Example(features=tf.train.Features(feature={
            'channels': int64_feature(self.channels),
            'width': int64_feature(self.width),
            'height': int64_feature(self.height),
            'name': bytes_feature(self.name.encode()),
            'left': bytes_feature(self.left.tostring()),
            'right': bytes_feature(self.right.tostring()),
            'disparity': bytes_feature(self.disparity.tostring()),
        }))


class KittiExample(BaseExample):
    width = 1244
    height = 376

    def __init__(self, path):
        super().__init__(path)
        self.left = padding(open_image(path['left']), self.size)[None, :, :, :]
        self.right = padding(open_image(path['right']), self.size)[None, :, :, :]
        self.disparity = padding(load_disparity(path['disparity']), self.size[:-1])[None, :, :, None]


class KittiOdometryExample(BaseExample):
    width = 1244
    height = 376

    def __init__(self, path):
        super().__init__(path)
        left = open_image(path['left'])
        self.original_shape = left.shape
        self.left = padding(left, self.size)[None, :, :, :]
        self.right = padding(open_image(path['right']), self.size)[None, :, :, :]

    def to_tfrecords(self):
        return tf.train.Example(features=tf.train.Features(feature={
            'channels': int64_feature(self.channels),
            'width': int64_feature(self.width),
            'height': int64_feature(self.height),
            'name': bytes_feature(self.name.encode()),
            'left': bytes_feature(self.left.tostring()),
            'right': bytes_feature(self.right.tostring()),
        }))


class SceneFlowExample(BaseExample):
    width = 960
    height = 540

    def __init__(self, path):
        super().__init__(path)
        self.left = open_image(path['left'])[None, :, :, :-1]
        self.right = open_image(path['right'])[None, :, :, :-1]
        self.disparity = load_pfm(path['disparity'], self.height, self.width).astype('float32')[None, :, :, None]


example_classes = {
    'kitti': KittiExample,
    'kitti_submission': KittiExample,
    'sceneflow': SceneFlowExample,
    'odometry': KittiOdometryExample,
}


def get_example_class(name):
    return example_classes[name]
