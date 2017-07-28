import tensorflow as tf
import os
import sys

from data.tfrecords import read_and_decode
from data.decoders import get_decoder_class
from data.shapes import InputShape
from collections import namedtuple
from experiment.configuration import Configuration
from experiment.logger import Logger
from models.factory import create_model

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

flags.DEFINE_string('config', None, 'Configuration file')

FLAGS = flags.FLAGS


def get_decoder_configurations(flags, config):
    shapes = InputShape(flags.width, flags.height, 3, flags.max_disp, 256)
    shapes_l = InputShape(900, 300, 3, flags.max_disp, 256)
    train_size = int(round(flags.examples * flags.train_ratio / flags.batch_size))
    train_valid_size = int(round(flags.examples * flags.train_valid_ratio / flags.batch_size))
    valid_size = int(round(flags.examples * flags.valid_ratio / flags.batch_size))
    # test_size = int(round(flags.examples * flags.test_ratio / flags.batch_size))
    return [
        DecodeConfig('train', flags, True, train_size, shapes, config.train),
        DecodeConfig('train_valid', flags, False, train_valid_size, shapes_l, config.train_valid),
        DecodeConfig('valid', flags, False, valid_size, shapes_l, config.valid),
        # DecodeConfig('test', flags, False, test_size, None, config.test),
    ]


def main(_):
    model_config = Configuration(FLAGS.config)
    configs = get_decoder_configurations(FLAGS, model_config)
    decoder_class = get_decoder_class(FLAGS.dataset)
    with tf.variable_scope('placeholders'):
        placeholders = {}
        for config in configs:
            placeholders[config.name] = read_and_decode(
                tf.train.string_input_producer(config.queues, shuffle=config.is_training, capacity=FLAGS.capacity,
                                               name='input_{}'.format(config.name)), decoder_class(config))
    model = create_model(FLAGS, model_config)
    model.build(placeholders['train'], True, None)
    model.build(placeholders['train_valid'], False, True)
    model.build(placeholders['valid'], False, True)
    saver = tf.train.Saver()
    session = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(FLAGS.lr).minimize(model.losses[placeholders['train']])
    session.run(tf.local_variables_initializer())
    session.run(tf.global_variables_initializer())
    if model_config.checkpoint is not None:
        saver.restore(session, model_config.checkpoint)
    f = open(os.path.join(model_config.directory, 'log.txt'), 'w')
    sys.stdout = Logger(sys.stdout, f)
    checkpoints = os.path.join(model_config.directory, 'checkpoints')
    os.makedirs(checkpoints, exist_ok=True)
    train_epoch_steps = int(FLAGS.examples * FLAGS.train_ratio / FLAGS.batch_size)
    train_valid_epoch_steps = int(FLAGS.examples * FLAGS.valid_ratio / FLAGS.batch_size)
    valid_epoch_steps = int(FLAGS.examples * FLAGS.train_valid_ratio / FLAGS.batch_size)
    try:
        for epoch in range(FLAGS.epochs):
            for _ in range(train_epoch_steps):
                _, train_loss = session.run([train_step, model.losses[placeholders['train']]])
                print("train: epoch {} loss {}".format(epoch, train_loss))
            for _ in range(train_valid_epoch_steps):
                valid_loss = session.run(model.losses[placeholders['valid']])
                print("valid: epoch {} loss {}".format(epoch, valid_loss))
            if train_valid_epoch_steps > 0:
                train_valid_losses = []
                for _ in range(valid_epoch_steps):
                    train_valid_losses.append(session.run(model.losses[placeholders['train_valid']]))
                    print("train_valid: epoch {} loss {}".format(epoch, train_valid_losses[-1]))
                current = sum(train_valid_losses) / len(train_valid_losses)
                if epoch == 0:
                    best = current
                if current <= best:
                    saver.save(session, os.path.join(checkpoints, '{}.cpkt'.format(epoch)))

    except Exception as e:
        print(e)
    finally:
        saver.save(session, os.path.join(checkpoints, 'final.cpkt'))
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
