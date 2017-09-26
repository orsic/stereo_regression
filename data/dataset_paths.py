import os
import itertools
import random
from collections import namedtuple

import config

TF_RECORDS_HOME = config.get('TF_RECORDS_HOME')

SCENE_FLOW_HOME = config.get('SCENE_FLOW_HOME')


def get_driving_paths():
    datasets = [os.path.join(focal, side, speed) for focal, side, speed in
                itertools.product(
                    ['15mm_focallength', '35mm_focallength'],
                    ['scene_forwards', 'scene_backwards'],
                    ['slow', 'fast'])]
    paths = []
    for dataset in datasets:
        dataset_path = os.path.join(SCENE_FLOW_HOME, 'driving/frames_finalpass', dataset)
        disparity_path = os.path.join(SCENE_FLOW_HOME, 'driving_disparity/disparity', dataset)
        dataset_examples = [example[:-4] for example in
                            filter(lambda x: x.endswith('png'),
                                   os.listdir(os.path.join(dataset_path, 'left')))]
        paths.extend([{'left': os.path.join(dataset_path, 'left', example + '.png'),
                       'right': os.path.join(dataset_path, 'right', example + '.png'),
                       'disparity': os.path.join(dataset_path, disparity_path, 'left', example + '.pfm')}
                      for example in dataset_examples])
    return paths


def get_monkaa_paths():
    home = os.path.join(SCENE_FLOW_HOME, 'monkaa/frames_finalpass')
    disparity = os.path.join(SCENE_FLOW_HOME, 'monkaa_disparity/disparity')
    paths = []
    datasets = os.listdir(home)
    for dataset in datasets:
        examples = filter(lambda x: x.endswith('.png'), os.listdir(os.path.join(home, dataset, 'left')))
        dataset_examples = [example[:-4] for example in examples]
        paths.extend([{
            'left': os.path.join(home, dataset, 'left', example + '.png'),
            'right': os.path.join(home, dataset, 'right', example + '.png'),
            'disparity': os.path.join(disparity, dataset, 'left', example + '.pfm'),
        } for example in dataset_examples])
    return paths


def get_chairs_paths():
    home = os.path.join(SCENE_FLOW_HOME, 'flyingthings3d/frames_finalpass/TRAIN')
    disparity = os.path.join(SCENE_FLOW_HOME, 'flyingthings3d_disparity/disparity/TRAIN')
    paths = []
    for side in ['A', 'B', 'C']:
        seqs = os.listdir(os.path.join(home, side))
        for seq in seqs:
            examples = filter(lambda x: x.endswith('.png'), os.listdir(os.path.join(home, side, seq, 'left')))
            dataset_examples = [example[:-4] for example in examples]
            paths.extend([{
                'left': os.path.join(home, side, seq, 'left', example + '.png'),
                'right': os.path.join(home, side, seq, 'right', example + '.png'),
                'disparity': os.path.join(disparity, side, seq, 'left', example + '.pfm'),
            } for example in dataset_examples])
    return paths


def get_all_sceneflow_paths():
    paths = []
    paths += get_driving_paths()
    paths += get_monkaa_paths()
    paths += get_chairs_paths()
    return paths


KITTI_HOME = config.get('KITTI_HOME')
KITTI_TRAINING_HOME = os.path.join(KITTI_HOME, 'training')
KITTI_TRAINING_LEFT = os.path.join(KITTI_TRAINING_HOME, 'image_2')
KITTI_TRAINING_RIGHT = os.path.join(KITTI_TRAINING_HOME, 'image_3')
KITTI_TRAINING_DISPARITIES = os.path.join(KITTI_TRAINING_HOME, 'disp_occ_0')


def get_all_kitti_paths():
    random.seed(5)
    paths = [{
        'left': os.path.join(KITTI_TRAINING_LEFT, image_name),
        'right': os.path.join(KITTI_TRAINING_RIGHT, image_name),
        'disparity': os.path.join(KITTI_TRAINING_DISPARITIES, image_name),
    } for image_name in sorted(os.listdir(KITTI_TRAINING_DISPARITIES))]
    random.shuffle(paths)
    return paths


KITTI_ODOMETRY_HOME = config.get('KITTI_ODOMETRY_HOME')


def get_odometry_paths():
    random.seed(5)
    paths = []
    for sequence in sorted(os.listdir(KITTI_ODOMETRY_HOME)):
        left_folder = os.path.join(KITTI_ODOMETRY_HOME, sequence, 'image_2')
        right_folder = os.path.join(KITTI_ODOMETRY_HOME, sequence, 'image_3')
        for name in sorted(filter(lambda x: x.endswith('png'), os.listdir(left_folder))):
            paths.append({
                'disparity': None,
                'left': os.path.join(left_folder, name),
                'right': os.path.join(right_folder, name),
            })
    random.shuffle(paths)
    return paths


KITTI_TESTING_HOME = os.path.join(KITTI_HOME, 'testing')
KITTI_TESTING_LEFT = os.path.join(KITTI_TESTING_HOME, 'image_2')
KITTI_TESTING_RIGHT = os.path.join(KITTI_TESTING_HOME, 'image_3')

CITYSCAPES_HOME = config.get('CITYSCAPES_HOME')
CITYSCAPES_LEFT = os.path.join(CITYSCAPES_HOME, 'left/leftImg8bit')
CITYSCAPES_RIGHT = os.path.join(CITYSCAPES_HOME, 'right/rightImg8bit')

CityscapesPath = namedtuple('CityscapesPath', 'split city name left right')


def get_cityscapes_paths():
    def getp(split, city, file):
        name = file.split('.')[0]
        file_r = file.replace('left', 'right')
        L, R = os.path.join(CITYSCAPES_LEFT, split, city, file), os.path.join(CITYSCAPES_RIGHT, split, city, file_r)
        return CityscapesPath(split, city, name, L, R)

    splits = os.listdir(CITYSCAPES_LEFT)
    paths = {}
    for split in splits:
        paths[split] = []
        cities = os.listdir(os.path.join(CITYSCAPES_LEFT, split))
        for city in cities:
            filenames = os.listdir(os.path.join(CITYSCAPES_LEFT, split, city))
            for filename in filenames:
                paths[split].append(getp(split, city, filename))
    return paths


def get_all_kitti_test_paths():
    paths = [{
        'left': os.path.join(KITTI_TESTING_LEFT, image_name),
        'right': os.path.join(KITTI_TESTING_RIGHT, image_name),
        'disparity': None,
    } for image_name in sorted(filter(lambda x: x.endswith('png'), os.listdir(KITTI_TESTING_LEFT)))]
    random.shuffle(paths)
    return paths


def get_dummy_path():
    return get_all_kitti_paths()[:1]


def get_paths_for_dataset(name):
    creators = {
        'kitti': get_all_kitti_paths,
        'kitti_submission': get_all_kitti_test_paths,
        'sceneflow': get_all_sceneflow_paths,
        'odometry': get_odometry_paths,
        'cityscapes': get_cityscapes_paths,
        'dummy': get_dummy_path,
    }
    return creators[name]()
