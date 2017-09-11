import tensorflow as tf


def create_meshgrid(N, H, W):
    '''
    Create mesh of elements 1 .. H * W reshaped to tensor and duplicated N times
    :param N:
    :param H:
    :param W:
    :return:
    '''
    mw = tf.reshape(tf.range(W * H), (H, W))
    return tf.tile(tf.expand_dims(mw, 0), [N, 1, 1])


def shift_disparity(grid, disparity, direction):
    '''
    Perform disparity shift and return indices
    :param grid:
    :param disparity:
    :param direction:
    :return:
    '''
    return tf.clip_by_value(grid + direction * tf.squeeze(disparity, -1), 0, tf.size(disparity) - 1)


def reconstruct(input_image, disparity, target_image='L'):
    '''
    Performs differentiable linear reconstruction of image
    :param left_image tf.Tensor of (N,H,W,C):
    :param disparity: tf.Tensor of (N,H,W,1):
    :param target_image:
    :return: tf.Tensor of (N,H,W,C)
    '''
    assert target_image in ['L', 'R']
    # determine direction to shift disparities
    shift = tf.constant(-1 if target_image == 'L' else 1, tf.int32)
    # get image shape
    image_shape = tf.shape(input_image)
    # get channel dims
    N, H, W, C = image_shape[0], image_shape[1], image_shape[2], image_shape[3]
    # channel by channel
    channels = tf.unstack(input_image, axis=-1)
    channel_transforms = []
    # create mesh of elements 1 .. H * W reshaped to tensor and duplicated N times
    index_grid = create_meshgrid(N, H, W)
    # flatten disparity
    disp_flat = tf.reshape(disparity, [-1])
    # get floor of disparity
    dint_floor = tf.cast(tf.floor(disparity), tf.int32)
    # get ceil of disparity
    dint_ceil = tf.cast(tf.ceil(disparity), tf.int32)
    # get shifted indices as 1D
    index_shift_floor = tf.reshape(shift_disparity(index_grid, dint_floor, shift), [-1], name='reshape_index_floor')
    index_shift_ceil = tf.reshape(shift_disparity(index_grid, dint_ceil, shift), [-1], name='reshape_index_ceil')
    for i, channel in enumerate(channels):
        with tf.variable_scope('channel_{}'.format(i)):
            # flatten channel
            flat = tf.reshape(channel, [-1])
            # caluclate interpolation weights
            alpha = 1.0 - (disp_flat - tf.cast(tf.reshape(dint_floor, [-1]), tf.float32))
            # reconstruct flattened
            rec_flat_floor = tf.cast(tf.gather(flat, index_shift_floor), tf.float32, name='gather_floor')
            rec_flat_ceil = tf.cast(tf.gather(flat, index_shift_ceil), tf.float32, name='gather_ceil')
            # interpolate recontruction
            rec_flat = rec_flat_floor * alpha + rec_flat_ceil * (1.0 - alpha)
            # reshape batch for current channel
            channel_transforms.append(tf.reshape(rec_flat, (N, H, W)))
    return tf.stack(channel_transforms, axis=-1)
