import numpy as np


def disp_precision(disparity_true, disparity_net, max_disp, err_max):
    hit_indexes = (np.abs(disparity_net - disparity_true) <= err_max) & (disparity_true > 0) & (
        disparity_true <= max_disp)
    miss_indexes = (np.abs(disparity_net - disparity_true) > err_max) & (disparity_true > 0) & (
        disparity_true <= max_disp)
    hits = np.sum(hit_indexes)
    total = hits + np.sum(miss_indexes)
    return hits, total
