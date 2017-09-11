import numpy as np
from PIL import Image
import png


def load_disparity(path):
    disparity = np.array(Image.open(path)) / 256.0
    assert disparity is not None
    if len(disparity.shape) > 2:
        disparity = disparity[:, :, 0]
    return disparity.astype('float32')


def store_disparity(disparity, path):
    result = (disparity * 256.0).astype('uint16')
    writer = png.Writer(size=result.shape[::-1], greyscale=True, bitdepth=16)
    with open(path, 'wb') as fp:
        writer.write(fp, result)


def padding(img, size):
    assert (len(size) == 3 or len(size) == 2) and (len(img.shape) == len(size))
    zeros = np.zeros(size, dtype=img.dtype)
    if len(size) == 2:
        zeros[:img.shape[0], :img.shape[1]] = img
    else:
        zeros[:img.shape[0], :img.shape[1], :] = img
    return zeros


def open_image(path):
    img = np.array(Image.open(path))
    return (2 * (img / img.max()) - 1.0).astype('float32')


def load_pfm(filename, height, width):
    file = open(filename, 'rb')

    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    else:
        color = False

    file.readline()

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def split_dataset_paths(paths, train_ratio, train_valid_ratio, valid_ratio, test_ratio):
    assert np.isclose(np.sum([train_ratio, train_valid_ratio, valid_ratio, test_ratio]), 1.0)
    size = len(paths)
    train_size = int(train_ratio * size)
    train_valid_size = int(train_valid_ratio * size)
    valid_size = int(valid_ratio * size)
    test_size = int(test_ratio * size)
    begin, end = 0, train_size
    train_paths = paths[begin:end]
    begin, end = end, end + train_valid_size
    train_valid_paths = paths[begin:end]
    begin, end = end, end + valid_size
    valid_paths = paths[begin:end]
    begin, end = end, end + test_size
    test_paths = paths[begin:end]
    return {
        'train': train_paths,
        'train_valid': train_valid_paths,
        'valid': valid_paths,
        'test': test_paths,
    }
