import numpy as np

ORIENTATION = {
    'coronal': "COR",
    'axial': "AXI",
    'sagital': "SAG"
}

def remove_black(data, labels):
    clean_data, clean_labels = [], []
    for i, frame in enumerate(labels):
        unique, counts = np.unique(frame, return_counts=True)
        if counts[0] / sum(counts) < .99:
            clean_labels.append(frame)
            clean_data.append(data[i])
    return np.array(clean_data), np.array(clean_labels)
    
def remove_black_3channels(data,fat,water, labels, inv, return_indices=False):
    clean_data,clean_fat,clean_water, clean_labels, clean_in = [], [],[],[], []
    start, end = None, -1
    for i, frame in enumerate(labels):
        unique, counts = np.unique(frame, return_counts=True)
        if counts[0] / sum(counts) < .99:
            if start is None:
                start = i
            if end < i:
                end = i
            clean_labels.append(frame)
            clean_data.append(data[i])
            if water is not None:
                clean_water.append(water[i])
            if fat is not None:
                clean_fat.append(fat[i])
            if inv is not None:
                clean_in.append(inv[i])
    if return_indices:
        return np.array(clean_data), np.array(clean_fat), np.array(clean_water), np.array(clean_labels), np.array(clean_in), start, end+1
    else:
        return np.array(clean_data), np.array(clean_fat), np.array(clean_water), np.array(clean_labels), np.array(clean_in)

def rotate_orientation(volume_data, orientation=ORIENTATION['coronal']):
    if orientation == ORIENTATION['axial']:
        return volume_data.transpose((2, 1, 0))
    elif orientation == ORIENTATION['sagital']:
        return volume_data
    elif orientation == ORIENTATION['coronal']:
        return volume_data.transpose((1,0,2))
    else:
        raise ValueError("Invalid value for orientation. Pleas see help")

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