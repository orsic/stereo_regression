import tensorflow as tf


class SoftArgmin():
    def __init__(self, flags, config):
        self.max_disp = config.get('max_disp', flags.max_disp)
        self._disp_mul = None
        self.outputs = {}

    def build(self, inputs):
        with tf.variable_scope('softargmin'):
            probs = tf.nn.softmax(-1 * inputs, dim=1, name='softmax')
            soft_argmin = tf.reduce_sum(probs * self.disp_mul, axis=1, name='soft_argmin_out', keep_dims=False)
        self.outputs[inputs] = soft_argmin

    @property
    def disp_mul(self):
        if self._disp_mul is None:
            dmul = tf.cast(tf.range(0, self.max_disp), tf.float32)
            self._disp_mul = tf.reshape(dmul, (1, self.max_disp, 1, 1, 1))
        return self._disp_mul
