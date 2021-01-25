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


def rotate_orientation(volume_data, orientation=ORIENTATION['coronal']):
    if orientation == ORIENTATION['axial']:
        return volume_data.transpose((2, 1, 0))
    elif orientation == ORIENTATION['sagital']:
        return volume_data
    elif orientation == ORIENTATION['coronal']:
        return volume_data.transpose((1,0,2))
    else:
        raise ValueError("Invalid value for orientation. Pleas see help")


def estimate_weights_mfb(labels):
    class_weights = np.zeros_like(labels)
    # TODO: don't hardcode this...
    unique = np.array([0,1,2,3,4,5,6,7,8])
    counts = np.zeros_like(unique)
    for l in unique:
        counts[l] = np.count_nonzero(labels==unique[l])

    #unique, counts = np.unique(labels, return_counts=True)
    #print('unique: ', unique)
    #print('counts: ', counts)
    median_freq = np.median(counts)
    weights = np.ones(9)
    # remove the gallbladder (since if this organ is missing in our dataset, it's because the patient doesn't
    # have a gallbladder. All other missing organs mean just missing segmentations
    tmp = counts.copy()
    tmp = np.delete(tmp, 8)
    #print('tmp after deleting gallbladder ', tmp)
    #print('counts still same? ', counts)
    if np.count_nonzero(tmp == 0) > 0: # this means we have missing annotations for one or more organs
        # set background counts to zero --> this will result in setting the weights for background to zero
        # as we have missing organ annotations we don't want the gradients for background to backprop as this
        # background label has false positives now
        counts[0] = 0

    for i, label in enumerate(unique):
        if counts[i] == 0: # this would lead to nan weights
            weights[int(label)] = 0
            class_weights += 0*np.array(labels == label)
        else:
            class_weights += (median_freq / counts[i]) * np.array(labels == label)
            #weights[int(label)] = median_freq / counts[i]
    #print('weights: ', weights)
    grads = np.gradient(labels)
    edge_weights = (grads[0] ** 2 + grads[1] ** 2) > 0
    class_weights += 2 * edge_weights
    return class_weights, weights


def reduce_slices_3channel(data, fat, water, labels, skip_Frame=40):
    """
    This function removes the useless black slices from the start and end. And then selects every even numbered frame.
    """
    no_slices, H, W = data.shape
    mask_vector = np.zeros(no_slices, dtype=int)
    mask_vector[::2], mask_vector[1::2] = 1, 0
    mask_vector[:skip_Frame], mask_vector[-skip_Frame:-1] = 0, 0

    data_reduced = np.compress(mask_vector, data, axis=0).reshape(-1, H, W)
    labels_reduced = np.compress(mask_vector, labels, axis=0).reshape(-1, H, W)
    fat_reduced = np.compress(mask_vector, fat, axis=0).reshape(-1, H, W)
    water_reduced = np.compress(mask_vector, water, axis=0).reshape(-1, H, W)


    return data_reduced, fat_reduced, water_reduced, labels_reduced


def reduce_slices_2channel(data, water, labels, skip_Frame=40):
    """
    This function removes the useless black slices from the start and end. And then selects every even numbered frame.
    """
    no_slices, H, W = data.shape
    mask_vector = np.zeros(no_slices, dtype=int)
    mask_vector[::2], mask_vector[1::2] = 1, 0
    mask_vector[:skip_Frame], mask_vector[-skip_Frame:-1] = 0, 0

    data_reduced = np.compress(mask_vector, data, axis=0).reshape(-1, H, W)
    labels_reduced = np.compress(mask_vector, labels, axis=0).reshape(-1, H, W)
    # fat_reduced = np.compress(mask_vector, fat, axis=0).reshape(-1, H, W)
    water_reduced = np.compress(mask_vector, water, axis=0).reshape(-1, H, W)

    return data_reduced, water_reduced, labels_reduced


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

def remove_black_3channels(data,fat,water, labels, inv, return_points=False):
    clean_data,clean_fat,clean_water, clean_labels, clean_in = [], [],[],[], []
    start, end = 0,0
    for i, frame in enumerate(labels):
        # print(i)
        unique, counts = np.unique(frame, return_counts=True)
        if counts[0] / sum(counts) < .99:
            if start == 0:
                start = i
            if i>end:
                end =i
            # print(start, end)
            clean_labels.append(frame)
            clean_data.append(data[i])
            clean_water.append(water[i])
            # clean_fat.append(fat[i])
            clean_in.append(inv[i])
    if return_points:
        return np.array(clean_data), np.array(clean_fat), np.array(clean_water), np.array(clean_labels), np.array(clean_in), start, end+1
    else:
        return np.array(clean_data), np.array(clean_fat), np.array(clean_water), np.array(clean_labels), np.array(clean_in)

def remove_black(data, labels):
    clean_data, clean_labels = [], []
    for i, frame in enumerate(labels):
        unique, counts = np.unique(frame, return_counts=True)
        if counts[0] / sum(counts) < .99:
            clean_labels.append(frame)
            clean_data.append(data[i])
    return np.array(clean_data), np.array(clean_labels)
