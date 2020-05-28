import numpy as np
from skimage.measure import regionprops

import numpy as np
import nibabel as nb
from nibabel.affines import from_matvec, to_matvec, apply_affine
import scipy.interpolate as si
import operator
import glob
import nrrd
import os
import numpy.linalg as npl



ORIENTATION = {
    'coronal': "COR",
    'axial': "AXI",
    'sagital': "SAG"
}


def rotate_orientation(volume_data, volume_label, orientation=ORIENTATION['coronal']):
    if orientation == ORIENTATION['axial']:
        return volume_data.transpose((2, 1, 0)), volume_label.transpose((2, 1, 0))
    elif orientation == ORIENTATION['sagital']:
        return volume_data, volume_label
    elif orientation == ORIENTATION['coronal']:
        return volume_data.transpose((1,0,2)), volume_label.transpose((1,0,2))
    else:
        raise ValueError("Invalid value for orientation. Pleas see help")


def estimate_weights_mfb(labels):
    class_weights = np.zeros_like(labels)
    unique, counts = np.unique(labels, return_counts=True)
    median_freq = np.median(counts)
    weights = np.zeros(79)
    for i, label in enumerate(unique):
        class_weights += (median_freq // counts[i]) * np.array(labels == label)
        weights[int(label)] = median_freq // counts[i]

    grads = np.gradient(labels)
    edge_weights = (grads[0] ** 2 + grads[1] ** 2) > 0
    class_weights += 2 * edge_weights
    return class_weights, weights


def reduce_slices(data, labels, skip_Frame=40):
    """
    This function removes the useless black slices from the start and end. And then selects every even numbered frame.
    """
    no_slices, H, W = data.shape
    mask_vector = np.zeros(no_slices, dtype=int)
    mask_vector[::2], mask_vector[1::2] = 1, 0
    mask_vector[:skip_Frame], mask_vector[-skip_Frame:-1] = 0, 0

    data_reduced = np.compress(mask_vector, data, axis=0).reshape(-1, H, W)
    labels_reduced = np.compress(mask_vector, labels, axis=0).reshape(-1, H, W)

    return data_reduced, labels_reduced


def remove_black(data, labels):
    clean_data, clean_labels = [], []
    for i, frame in enumerate(labels):
        unique, counts = np.unique(frame, return_counts=True)
        if counts[0] / sum(counts) < .99:
            clean_labels.append(frame)
            clean_data.append(data[i])
    return np.array(clean_data), np.array(clean_labels)
