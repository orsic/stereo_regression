import tensorflow as tf
from collections import namedtuple
import os
import sys

from clint.textui import progress

from models.factory import create_model
from experiment.configuration import Configuration
from experiment.util import disp_precision
from data.util import split_dataset_paths, store_disparity
from data.dataset import Dataset
from data.dataset_paths import get_paths_for_dataset
from data.example import get_example_class
from experiment.logger import Logger
from models.util import predict_strided

DecodeConfig = namedtuple('DecodeConfig', 'name flags is_training size shapes queues')

flags = tf.app.flags

flags.DEFINE_string('model', None, 'Name of the model to create')
flags.DEFINE_string('dataset', 'kitti', 'Name of the dataset to prepare')
flags.DEFINE_integer('epochs', 100, 'Number of train epochs')
flags.DEFINE_integer('examples', 200, 'Number of dataset examples')
flags.DEFINE_float('lr', 1e-3, 'Initial learning rate')

flags.DEFINE_float('train_ratio', 0.8, 'Train subset split size')
flags.DEFINE_float('train_valid_ratio', 0.01, 'Train valid subset split size')
flags.DEFINE_float('valid_ratio', 0.19, 'Valid subset split size')
flags.DEFINE_float('test_ratio', 0.0, 'Test subset split size')

flags.DEFINE_integer('batch_size', 1, 'Batch size')
flags.DEFINE_integer('num_threads', 5, 'Number of reading threads')
flags.DEFINE_integer('capacity', 50, 'Queue capacity')

flags.DEFINE_integer('width', 512, 'Crop width')
flags.DEFINE_integer('height', 256, 'Crop width')
flags.DEFINE_integer('max_disp', 192, 'Maximum possible disparity')

flags.DEFINE_bool('strided', False, 'Wether or not to predict disparities patchwise')

flags.DEFINE_string('config', None, 'Configuration file')

FLAGS = flags.FLAGS


def main(_):
    # create global configuration object
    model_config = Configuration(FLAGS.config)
    model = create_model(FLAGS, model_config)
    placeholders = {
        'l': tf.placeholder(tf.float32, (1, None, None, 3)),
        'r': tf.placeholder(tf.float32, (1, None, None, 3)),
        'd': tf.placeholder(tf.float32, (1, None, None, 1)),
    }
    x = {
        'l': tf.placeholder(tf.float32, (1, None, None, 3)),
        'r': tf.placeholder(tf.float32, (1, None, None, 3)),
        'd': tf.placeholder(tf.float32, (1, None, None, 1)),
    }
    p = namedtuple('Placeholders', placeholders.keys())(**placeholders)
    px = namedtuple('Placeholders', x.keys())(**x)
    model.build(px, True, None)
    model.build(p, False, True)
    session = tf.Session()
    saver = tf.train.Saver()
    # init variables
    session.run(tf.local_variables_initializer())
    session.run(tf.global_variables_initializer())
    # restore model if provided a checkpoint
    if model_config.checkpoint is not None:
        print("Restoring model from {}".format(model_config.checkpoint))
        saver.restore(session, model_config.checkpoint)
    # init dataset
    paths = get_paths_for_dataset(FLAGS.dataset)
    if 'test' not in paths:
        paths = {
            'train': [],
            'train_valid': [],
            'valid': [],
            'test': paths,
        }
    dataset = Dataset(get_example_class(FLAGS.dataset), paths, FLAGS.dataset)
    fd = lambda x: {p.l: x.left, p.r: x.right}
    reconstructions = os.path.join(model_config.directory, 'submission')
    os.makedirs(reconstructions, exist_ok=True)
    for generator in dataset:
        rec_dir = os.path.join(reconstructions, generator.name)
        os.makedirs(rec_dir, exist_ok=True)
        with progress.Bar(label=generator.name, expected_size=generator.length) as bar:
            for i, example in enumerate(generator.examples):
                bar.show(i)
                if FLAGS.strided:
                    d = predict_strided(session, model, example, p)
                else:
                    d = session.run(model.outputs[p], fd(example)).squeeze()
                store_disparity(d, os.path.join(rec_dir, '{}.png'.format(example.name)))


if __name__ == '__main__':
    tf.app.run()
