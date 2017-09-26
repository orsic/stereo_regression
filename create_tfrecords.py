import tensorflow as tf

from data.dataset import Dataset
from data.dataset_paths import get_paths_for_dataset
from data.example import get_example_class
from data.tfrecords import create_tfrecords
from data.util import split_dataset_paths

flags = tf.app.flags

flags.DEFINE_string('dataset', 'kitti', 'Name of the dataset to prepare')
flags.DEFINE_float('train_ratio', 0.8, 'Train subset split size')
flags.DEFINE_float('train_valid_ratio', 0.01, 'Train valid subset split size')
flags.DEFINE_float('valid_ratio', 0.19, 'Valid subset split size')
flags.DEFINE_float('test_ratio', 0.0, 'Test subset split size')

FLAGS = flags.FLAGS


def main(_):
    paths = get_paths_for_dataset(FLAGS.dataset)
    if type(paths) != dict:
        ratios = {
            'train_ratio': FLAGS.train_ratio,
            'train_valid_ratio': FLAGS.train_valid_ratio,
            'valid_ratio': FLAGS.valid_ratio,
            'test_ratio': FLAGS.test_ratio,
        }
        paths = split_dataset_paths(paths, **ratios)
    dataset = Dataset(get_example_class(FLAGS.dataset), paths, FLAGS.dataset)
    create_tfrecords(dataset)


if __name__ == '__main__':
    tf.app.run()
