import tensorflow as tf


class CostVolume():
    def __init__(self, flags, config):
        self.max_disp = config.get('max_disp', flags.max_disp)
        self.volumes = {}

    def build(self, placeholder):
        embedding_L, embedding_R = placeholder[0], placeholder[1]
        with tf.variable_scope('cost_volume'):
            volume = tf.stack(
                [tf.concat((embedding_L, tf.concat([
                    tf.slice(embedding_R, [0, 0, tf.shape(embedding_R)[2] - d, 0], [-1, -1, d, -1]),
                    tf.slice(embedding_R, [0, 0, 0, 0], [-1, -1, tf.shape(embedding_R)[2] - d, -1])
                ], axis=2)), axis=3) for d in range(self.max_disp // 2)],
                axis=1, name='cost_volume_tensor'
            )
        self.volumes[placeholder] = volume


class SubCostVolume():
    def __init__(self, flags, config):
        self.max_disp = config.get('max_disp', flags.max_disp)
        self.volumes = {}

    def build(self, placeholder):
        embedding_L, embedding_R = placeholder[0], placeholder[1]
        with tf.variable_scope('cost_volume'):
            volume = tf.stack(
                [(embedding_L - tf.concat([
                    tf.slice(embedding_R, [0, 0, tf.shape(embedding_R)[2] - d, 0], [-1, -1, d, -1]),
                    tf.slice(embedding_R, [0, 0, 0, 0], [-1, -1, tf.shape(embedding_R)[2] - d, -1])
                ], axis=2)) for d in range(self.max_disp // 2)],
                axis=1, name='cost_volume_tensor'
            )
        self.volumes[placeholder] = volume
