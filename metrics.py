import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import re

flags = tf.app.flags

flags.DEFINE_string('log', None, 'Log file')

FLAGS = flags.FLAGS


def main(_):
    metrics = {
        'train': {},
        'train_valid': {},
        'valid': {},
    }
    template = r'([a-z_]*):\Wepoch\W(\d*)\Wloss\W([\d\.]*)'
    with open(FLAGS.log, 'r') as file:
        for line in file:
            line = line.rstrip()
            line_search = re.search(template, line)
            phase = line_search.group(1)
            epoch = int(line_search.group(2))
            loss = float(line_search.group(3))
            phase_metric = metrics[phase]
            epoch_metric = phase_metric.get(epoch, [])
            epoch_metric.append(loss)
            phase_metric[epoch] = epoch_metric
    mean_epoch_losses = {
        'train': [],
        'train_valid': [],
        'valid': [],
    }
    plt.title(FLAGS.log)
    for phase in metrics:
        for epoch in sorted(metrics[phase].keys()):
            mean_epoch_losses[phase].append(np.mean(metrics[phase][epoch]))
    for phase in mean_epoch_losses:
        plt.plot(mean_epoch_losses[phase], label=phase)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    tf.app.run()
