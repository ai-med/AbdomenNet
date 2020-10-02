import operator
import os
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
import nrrd
import glob

from nibabel.orientations import axcodes2ornt, ornt_transform, inv_ornt_aff, flip_axis
from nibabel.affines import from_matvec, to_matvec, apply_affine
from nibabel.processing import resample_to_output, resample_from_to
import numpy.linalg as npl

# from nilearn.image import resample_img

DATASET = 'NAKO' # 'NAKO', 'UKB', 'KORANAKOUKB'
DEFAULT_FILE_TYPE = 'nifti'  # 'nerd'
TARGET_FILE_TYPE = 'nifti'
DEFAULT_ORIENTATION = 'RAS'
TARGET_RESOLUTION = [2,2,3]
DEFAULT_VIEW = ['Saggital', 'Coronal', 'Axial']
DEFAULT_REFERENCE_VIEW = 'Sagittal'
OPTIMIZATION = 'N4'  # Intensity, Min-Max, Fat-Water-Swap
IS_CROPPING = True
DEFAULT_WORLD_COORDS = [500, 500, 1000]
DEFAULT_OUTPUT_PATH = './temp'
DEFAULT_LINSPACE = 30

FILE_TO_LABEL_MAP =  {'BACKGROUND': 'background','LIVER': 'liver', 'SPLEEN': 'spleen','KIDNEY(RIGHT)':'kidney_r',
                      'KIDNEY(LEFT)':'kidney_l', 'ADRENALGLAND':'adrenal', 'PANCREAS': 'pancreas',
                      'GALLBLADDER': 'gallbladder', 'SUBCUTANEOUS':'subcutaneous', 'THYROIDGLAND':'thyroid_gland'}


def create_if_not(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Creating the default paths needed for smooth execution.
create_if_not(DEFAULT_OUTPUT_PATH)


def volume_viewer(vol, axis_idx=0):
    if axis_idx > 2:
        raise Exception('Axis Index cannot be more than 2! Ideally 0: Sagittal, 1: Coronal, 2: Axial.')
    axis = vol.shape
    plt.imshow(vol[axis[axis_idx] // 2])
    plt.show()


def volume_3_view_viewer(vol):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, tight_layout=True)
    axis = vol.shape

    ax1.imshow(vol[axis[0] // 2])
    ax2.imshow(vol[:, axis[1] // 2, :])
    ax3.imshow(vol[:, :, axis[2] // 2])

    plt.show()


def get_volume_data(img):
    print(f"Affine:{img.affine}, Image Shape: {img.shape}")
    return img.get_fdata()


def save_volume(img, file_name):
    nb.save(img, f'{file_name}.nii.gz')


def sigmoid(x):
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = 1 / (1 + np.exp(-x[i]))
    return y


def normalise_data(volume):
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    return volume




def post_interpolate(volume, labelmap=None, target_shape=[256,256,128]):
    volume = do_cropping(volume, target_shape)
    if labelmap is not None:
        labelmap = do_cropping(labelmap, target_shape)
    current_shape = volume.shape
    intended_shape_deficit = target_shape - np.asarray(current_shape)

    paddings = [tuple(
        np.array([np.ceil((pad_tuples / 2) - pad_tuples % 2), np.floor((pad_tuples / 2) + pad_tuples % 2)]).astype(
            'int32')) for pad_tuples in intended_shape_deficit]
    paddings = tuple(paddings)

    volume = np.pad(volume, paddings, mode='constant')
    if labelmap is not None:
        labelmap = np.pad(labelmap, paddings, mode='constant')

    return volume, labelmap

def do_cropping(source_num_arr, bounding):
    start = list(map(lambda a, da: a // 2 - da // 2, source_num_arr.shape, bounding))
    end = list(map(operator.add, start, bounding))
    for i, val in enumerate(zip(start, end)):
        if val[0] < 0:
            start[i] = 0
            end[i] = source_num_arr.shape[i]
    slices = tuple(map(slice, tuple(start), tuple(end)))
    return source_num_arr[slices]

def drop_overlapped_pixels(labelmap, availed_manual_segs_id_list, no_of_class):
    present_seg_idxs = np.unique(labelmap)
    overlapped_seg_idxs = set(present_seg_idxs).difference(availed_manual_segs_id_list)

    for idxs in overlapped_seg_idxs:
        print(f'Overlapped Idxs Found, removing it for idx {idxs}')
        labelmap[labelmap == idxs] = 0
    return labelmap

def fetch_class_labels_from_filemap(labelmap_path, file_labels):
    label_idx, label = None, None
    for lidx, file_label in enumerate(file_labels):
        if file_label in labelmap_path.replace(" ", "").upper():
            label_idx, label = lidx, file_label
            break
    return label_idx, label


def nrrd_reader(file_path):
    print("Reading NRRD Files.....")
    _nrrd = nrrd.read(file_path)
    data = _nrrd[0]
    header = _nrrd[1]
    return data, header, None


def nibabel_reader(file_path):
    print("Reading Nifti Files.....")
    volume_nifty = nb.load(file_path)
    volume = get_volume_data(volume_nifty)
    return volume, volume_nifty.header, volume_nifty


def file_reader(file_path, file_type=None):
    print('Reading Files.....')
    header_mat = np.empty_like((4, 4))
    if file_type == None:
        file_type = file_path.split('.')[-1]
    if file_type == 'nrrd':
        data, header, img = nrrd_reader(file_path)
        affine = header['space directions']
        affine = affine[:3, :3]
        origins = header['space origin']
        origins = origins[:3]
        t_mat = from_matvec(affine, origins)
        img = nb.Nifti1Image(data, t_mat) if img is None else img
        header_mat = t_mat
    else:
        data, header, img = nibabel_reader(file_path)
        header_mat = header

    return data, header_mat, img


def remove_black(volume):
    print("Removing Black Slices.....")
    clean_data = []
    for i, frame in enumerate(volume):
        unique, counts = np.unique(frame, return_counts=True)
        if counts[0] / sum(counts) < .99:
            clean_data.append(frame)
    return np.array(clean_data)


def do_nibabel_transform_to_ras(img):
    print(f'Transforming Images to {DEFAULT_ORIENTATION}.....')
    affine = img.affine
    orig_ornt = nb.io_orientation(affine)
    targ_ornt = axcodes2ornt(DEFAULT_ORIENTATION)
    transform = ornt_transform(orig_ornt, targ_ornt)
    img = img.as_reoriented(transform)
    return img


def multi_vol_stitching(images, is_label=False):
    if len(images) == 1:
        return images[0]
    elif len(images) == 0:
        raise Exception("Empty Image List!")

    images_sorted = sorted(images, key=lambda im: im.header['qoffset_z'], reverse=True)
    img_0 = images_sorted[0]

    mode = 'nearest' if is_label else 'constant'
    img_0 = resample_to_output(img_0, TARGET_RESOLUTION, order=3, mode=mode, cval=0.0)

    for idx, img_1 in enumerate(images_sorted[1:]):
        print(f'{idx}th img for stitching...')
        #         print("STARTTTTT: ", img_1.affine)
        img_1 = resample_to_output(img_1, TARGET_RESOLUTION, order=3, mode=mode, cval=0.0)
        #         img_1 = placing_axes(img_1, img_0.affine.copy(), img_0.header.copy(), [2])
        #         print( img_1.affine)
        target_affine = img_0.affine.copy()
        target_affine[2, 3] = img_1.affine[2, 3].copy()
        target_shape = img_0.shape[:2] + img_1.shape[2:]
        #         img_1 = placing_axes(img_1, target_affine, img_0.header.copy(), skip_axis=[2])
        img_1 = resample_from_to(img_1, [target_shape, target_affine])
        #         print(img_1.affine)
        img_0 = vol_stitching(img_0, img_1)
    #         print(img_0.affine, 'FINISHHHHHHHHH')

    return img_0


def vol_stitching(im_0, im_1):
    im_0_z = im_0.shape[2]
    im_1_z = im_1.shape[2]

    # calculate overlap region:
    im_0_end = im_0.header['qoffset_z']
    im_1_end = im_1.header['qoffset_z']

    spacing = im_0.header['pixdim'][3]

    im_0_width = im_0_z * spacing
    im_1_width = im_1_z * spacing

    im_1_start = im_1_end + im_1_width
    im_0_start = im_0_end + im_0_width

    overlap = abs(im_0_end - im_1_start)

    overlap_v = int(round(overlap / spacing))

    new_im_dim = abs(round((abs(im_1_end - im_0_start)) / spacing))

    new_img = np.empty([im_0.shape[0], im_0.shape[1], int(new_im_dim)])

    im_0_data = im_0.get_fdata()
    im_1_data = im_1.get_fdata()

    new_img[:, :, 0:(im_1_z - overlap_v)] = im_1_data[:, :, 0:(im_1_z - overlap_v)]
    new_img[:, :, im_1_z:] = im_0_data[:, :, overlap_v:]

    # overlap region:
    sigmoid_c = sigmoid(np.linspace(-DEFAULT_LINSPACE, DEFAULT_LINSPACE, overlap_v))

    for l in range(0, overlap_v):
        new_img[:, :, (im_1_z - overlap_v + l)] = \
            (1 - sigmoid_c[l]) * im_1_data[:, :, (im_1_z - overlap_v) + l] + (sigmoid_c[l]) * im_0_data[:, :, l]

    stitched_img = nb.Nifti1Image(new_img, im_1.affine, im_1.header)
    #     placing_axes(stitched_img)
    return stitched_img




def read_ras(file_path, file_type=None, is_label=False):
    _, _, img = file_reader(file_path, file_type)
#     print("header", img.header)
    img_ras = do_nibabel_transform_to_ras(img)
    if is_label:
        file_labels = list(FILE_TO_LABEL_MAP.keys())
        lidx, labelname = fetch_class_labels_from_filemap(file_path, file_labels)
        return img_ras, lidx, labelname
    else:
        return img_ras


def fetch_class_labels_from_filemap(labelmap_path, file_labels):
    label_idx, label = None, None
    for lidx, file_label in enumerate(file_labels):
        if file_label in labelmap_path.replace(" ", "").upper():
            label_idx, label = lidx, file_label
            break
    return label_idx, label


def placing_axes(vol, target_affine, target_header=None, skip_axis=None):
    vol2target = npl.inv(target_affine).dot(vol.affine)
    source_data = vol.get_fdata()
    shifts = tuple(vol2target[:3, 3].astype(np.int32))
#     print(shifts)
    #     print(source_data.shape)
    for ax, shift in enumerate(shifts):
        if skip_axis is not None and ax in skip_axis:
            continue
        print(ax, shift)
        shift = int(shift)
        if shift < 0:
            source_data = flip_axis(source_data, axis=ax)
        print(-np.abs(shift))
        source_data = np.roll(source_data, -np.abs(shift), axis=ax)

    if target_header is None:
        target_header = nb.Nifti1Header()
    stitched_labeled_img = nb.Nifti1Image(source_data, target_affine, target_header)

    return stitched_labeled_img





def hist_match(volume, histogram_matching_reference_path):
    template_file = nb.load(histogram_matching_reference_path)
    template = template_file.get_fdata()
    oldshape = volume.shape
    source = volume.ravel()
    template = template.ravel()

    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def estimate_weights_mfb(labels, no_of_class=8):
    class_weights = np.zeros_like(labels)
    unique, counts = np.unique(labels, return_counts=True)
    median_freq = np.median(counts)
    weights = np.zeros(no_of_class)
    for i, label in enumerate(unique):
        class_weights += (median_freq // counts[i]) * np.array(labels == label)
        weights[int(label)] = median_freq // counts[i]

    grads = np.gradient(labels)
    edge_weights = (grads[0] ** 2 + grads[1] ** 2) > 0
    class_weights += 2 * edge_weights

    return class_weights, weights


def estimate_weights_per_slice(labels, no_of_class=8):
    weights_per_slice = []
    for slice_ in labels:
        unique, counts = np.unique(slice_, return_counts=True)
        median_freq = np.median(counts)
        weights = np.zeros(no_of_class)
        for i, label in enumerate(unique):
            weights[int(label)] = median_freq // counts[i]
        weights_per_slice.append(weights)

    return np.array(weights_per_slice)



def label_stitch_extd(images, is_label=False):
    if len(images) == 1:
        return images
    elif len(images) == 0:
        raise Exception("Empty Image List!")

#     images_sorted = sorted(images, key=lambda im: im.header['qoffset_z'], reverse=True)
#     img_0 = images_sorted[0]

#     mode = 'nearest' if is_label else 'constant'
#     img_0 = resample_to_output(img_0, TARGET_RESOLUTION, order=3, mode=mode, cval=0.0)
    
    processed_segm = None #np.zeros_like(img_0.get_data())
    reference_labelmap = None
    target_affine = images[0][0].affine
    mode = 'nearest'
    for im_1, lidx, labelname in images:
        print(im_1.shape, lidx, labelname)
        if reference_labelmap is None:
            reference_labelmap = im_1
        else:
            im_1 = resample_from_to(im_1, [reference_labelmap.shape, reference_labelmap.affine], mode=mode)
            
        print(im_1.shape, lidx, labelname)    
    
        im_1_x, im_1_y, im_1_z = im_1.shape
        
        im_1_start_width_x = abs(im_1.header['qoffset_x'])
        im_1_start_width_y = abs(im_1.header['qoffset_y'])
        im_1_start_width_z = abs(im_1.header['qoffset_z'])

        spacing_img_1_x, spacing_img_1_y, spacing_img_1_z = im_1.header['pixdim'][1:4]
        print(spacing_img_1_x, spacing_img_1_y, spacing_img_1_z)

        im_1_width_x = im_1_x * spacing_img_1_x
        im_1_width_y = im_1_y * spacing_img_1_y
        im_1_width_z = im_1_z * spacing_img_1_z

        im_1_end_width_x = im_1_start_width_x + im_1_width_x
        im_1_end_width_y = im_1_start_width_y + im_1_width_y
        im_1_end_width_z = im_1_start_width_z + im_1_width_z
        
        im_1_end_x = im_1_end_width_x // spacing_img_1_x
        im_1_end_y = im_1_end_width_y // spacing_img_1_y
        im_1_end_z = im_1_end_width_z // spacing_img_1_z
        
        im_1_start_x = im_1_start_width_x // spacing_img_1_x
        im_1_start_y = im_1_start_width_y // spacing_img_1_y
        im_1_start_z = im_1_start_width_z // spacing_img_1_z

        im_1_data = im_1.get_fdata()
        
        im_1_data = np.multiply(lidx, im_1_data)
        
        im_1_start_x,im_1_end_x, im_1_start_y,im_1_end_y, im_1_start_z,im_1_end_z = int(im_1_start_x),int(im_1_end_x), int(im_1_start_y),int(im_1_end_y), int(im_1_start_z),int(im_1_end_z)
        print(im_1_start_x,im_1_end_x, im_1_start_y,im_1_end_y, im_1_start_z,im_1_end_z)
        if processed_segm is None:
            processed_segm = np.zeros((im_1_end_x, im_1_end_y, im_1_end_z))
#             print(processed_segm.shape)
        
        processed_segm[int(im_1_start_x):int(im_1_end_x), int(im_1_start_y):int(im_1_end_y), int(im_1_start_z):int(im_1_end_z)] += im_1_data
    
    labelmap = np.round(processed_segm)
    
    empty_header = nb.Nifti1Header()
    s_labelmap = nb.Nifti1Image(labelmap, target_affine, empty_header)
    
    return None, s_labelmap



