import os
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
import nrrd
import glob
import matplotlib.pyplot as plt

import SimpleITK as sitk
import sys

from nibabel.orientations import axcodes2ornt, ornt_transform, inv_ornt_aff, flip_axis
from nibabel.affines import from_matvec, to_matvec, apply_affine
from nibabel.processing import resample_to_output, resample_from_to
import numpy.linalg as npl

import torch
import torch.utils.data as data

from global_vars import *


def read_ras(file_path, file_type=None, is_label=False):
    _, _, img = file_reader(file_path, file_type)
    if img is None:
        return None if is_label is False else None, None, None
    img_ras = do_nibabel_transform_to_ras(img)
    if is_label:
        file_labels = list(FILE_TO_LABEL_MAP.keys())
        lidx, labelname = fetch_class_labels_from_filemap(file_path, FILE_TO_LABEL_MAP)
        return img_ras, lidx, labelname
    else:
        return img_ras
    
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
    elif file_type == 'gz' or file_type == 'nii':
        data, header, img = nibabel_reader(file_path)
        header_mat = header
    else:
        print(f'Unknown file type: {file_type} for file path: {file_path}')
        data, header_mat, img = None, None, None

    return data, header_mat, img

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

def do_nibabel_transform_to_ras(img):
    print(f'Transforming Images to {DEFAULT_ORIENTATION}.....')
    affine = img.affine
    orig_ornt = nb.io_orientation(affine)
    targ_ornt = axcodes2ornt(DEFAULT_ORIENTATION)
    transform = ornt_transform(orig_ornt, targ_ornt)
    img = img.as_reoriented(transform)
    return img

def fetch_class_labels_from_filemap(labelmap_path, file_labels=FILE_TO_LABEL_MAP):
    other_file = ['COMB']
    label_idx, label = None, None
    for lidx, file_label_key in enumerate(file_labels.keys()):
        ifPresent = np.any([label.replace(" ", "").upper() in labelmap_path.replace(" ", "").upper() for label in file_labels[file_label_key]])
        if ifPresent:
            label_idx, label = lidx, file_label_key
            break
    else:
        ifExceptedFilePresent = np.any([other.replace(" ", "").upper() in labelmap_path.replace(" ", "").upper() for other in other_file])
        if not ifExceptedFilePresent:
#             raise Exception(f'No Matched Label Found for {labelmap_path}!')
            print(f"Other labelmap values Found for {labelmap_path}")
        else:
            print(f"Other File Found for {labelmap_path}")
    return label_idx, label

def rescale(in_image, vol_id, original_filename):
    new_filename = original_filename.split('/')[-1].split('.')[0]
    in_image_data = in_image.get_fdata()
    o_min = np.min(in_image_data)
    o_max = np.max(in_image_data)
    print(o_min)
    if o_min<0:
        print('neagtive value detected')
        in_image_data_scaled = in_image_data + np.abs(o_min)+10
    else:
        in_image_data_scaled = in_image_data + 10
    
    u_min = np.min(in_image_data_scaled)
    u_max = np.max(in_image_data_scaled)
    
    in_image_scaled = nb.Nifti1Image(in_image_data_scaled, in_image.affine.copy(), in_image.header.copy())
    create_if_not(f'{n4_corrected_data_dir}/vol/{vol_id}')
    save_volume(in_image_scaled, f'{n4_corrected_data_dir}/vol/{vol_id}/{new_filename}_n4_scaled')

    return dict(
        SCALED= f'{n4_corrected_data_dir}/vol/{vol_id}/{new_filename}_n4_scaled.nii.gz',
        IN_BIAS= f'{n4_corrected_data_dir}/vol/{vol_id}/{new_filename}_n4_scaled_bias_field.nii.gz',
        IN_CORRECTED= f'{n4_corrected_data_dir}/vol/{vol_id}/{new_filename}_n4_scaled_corrected.nii.gz',
        MIN= u_min,
        MAX= u_max
    )

def SITK_N4_normalization(in_input_file, opp_file, output_file, shrink_factor=3,
                          mask=None, iteration=500, fittingLevel=4, tolerance=1e-03,
                          spline_param=200):
    inputImage = sitk.ReadImage(in_input_file, sitk.sitkFloat32)
    oppImage = sitk.ReadImage(opp_file, sitk.sitkFloat32)
    image = inputImage
    maskImage = sitk.OtsuThreshold(inputImage, 0, 1,200)

    if shrink_factor != None:
        image = sitk.Shrink(inputImage,
                                 [int(shrink_factor)] * inputImage.GetDimension())
        maskImage = sitk.Shrink(maskImage,
                                [int(shrink_factor)] * inputImage.GetDimension())
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    numberFittingLevels = 4
    if fittingLevel != None:
        numberFittingLevels = int(fittingLevel)
    if iteration != None:
        corrector.SetMaximumNumberOfIterations([int(iteration)]
                                               * numberFittingLevels)
    corrector.SetConvergenceThreshold(tolerance)
    
    low_res_output = corrector.Execute(image, maskImage)
    log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)
    output = oppImage / sitk.Exp( log_bias_field )
    sitk.WriteImage(output, output_file)
    print('done')
    return output_file

def create_if_not(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def save_volume(img, file_name):
    save_dir = '/'.join(file_name.split('/')[:-1])
    print("saving directory:", save_dir)
    create_if_not(save_dir)
    nb.save(img, f'{file_name}.nii.gz')

def get_volume_data(img):
    print(f"Affine:{img.affine}, Image Shape: {img.shape}")
    return img.get_fdata()
    
def apply_bias_field(opp_img, bias_field_img, opp_file, n4_dict, vol_id):
    new_filename = opp_file.split('/')[-1].split('.')[0]
    
    opp_img_data = opp_img.get_fdata()
    bias_field_img_data = bias_field_img.get_fdata()
    corrected_img_data = opp_img_data / bias_field_img_data
    
    orig_min = n4_dict['MIN']
    orig_max = n4_dict['MAX']
    current_min = np.min(corrected_img_data)
    current_max = np.max(corrected_img_data)
    downscaled_img_data = (corrected_img_data - current_min) * (orig_max - orig_min)/(current_max - current_min) + orig_min

    
    downscaled_img = nb.Nifti1Image(downscaled_img_data, opp_img.affine.copy(), opp_img.header.copy())
    create_if_not(f'{n4_corrected_data_dir}/vol/{vol_id}')

    save_volume(downscaled_img, f'{n4_corrected_data_dir}/vol/{vol_id}/{new_filename}_n4_corrected')
    n4_dict['OPP_CORRECTED'] = f'{n4_corrected_data_dir}/vol/{vol_id}/{new_filename}_n4_corrected.nii.gz'
    return n4_dict

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
        img_1 = resample_to_output(img_1, TARGET_RESOLUTION, order=3, mode=mode, cval=0.0)
        target_affine = img_0.affine.copy()
        target_affine[2, 3] = img_1.affine[2, 3].copy()
        target_shape = img_0.shape[:2] + img_1.shape[2:]
        img_1 = resample_from_to(img_1, [target_shape, target_affine])
        img_0 = vol_stitching(img_0, img_1)
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
    return stitched_img

def sigmoid(x):
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = 1 / (1 + np.exp(-x[i]))
    return y

def get_points(segm, img):
    if segm.shape == img.shape:
        return 0,0,0,segm.shape[0], segm.shape[1],segm.shape[2]
    img_dim_v = img.shape
    segm_dim_v = segm.shape
    print('im dim v: ', img_dim_v)
    print('segm dim v: ', segm_dim_v)

    img_h = img.header
    segm_h = segm.header
    im_spacing = abs(img_h['pixdim'][1:4])
    segm_spacing = abs(segm_h['pixdim'][1:4])
    print('im spacing: ', im_spacing)
    print('segm spacing: ', segm_spacing)

    im_dim_w = img_dim_v * im_spacing
    segm_dim_w = segm_dim_v * segm_spacing

    print('im dim w: ', im_dim_w)
    print('segm dim w: ', segm_dim_w)

    # correction of the wrong information in header file
    im_offx = abs(img_h['qoffset_x'])
    im_offy = abs(img_h['qoffset_y'])
    im_offz = img_h['qoffset_z']
    
    segm_offx = abs(segm_h['qoffset_x'])
    segm_offy = abs(segm_h['qoffset_y'])
    segm_offz = segm_h['qoffset_z']
    segm_off = np.array([segm_offx, segm_offy, segm_offz])

    im_offset = np.array([im_offx, im_offy, im_offz])
    im_start = im_offset
    im_end = np.array([im_start[0]-im_dim_w[0], im_start[1]-im_dim_w[1], im_start[2]+im_dim_w[2]])

    segm_start = segm_off
    segm_end = np.array([segm_start[0]-segm_dim_w[0], segm_start[1]-segm_dim_w[1], segm_start[2]+segm_dim_w[2]])

    print('im off: ', im_offset)
    print('segm off: ', segm_off)

    print('im start: ', im_start)
    print('im end: ', im_end)
    print('segm start: ', segm_start)
    print('segm end: ', segm_end)

    start_diff_w = abs(im_start - segm_start)
    end_diff_w = abs(im_end - segm_end)
    print('start diff w: ', start_diff_w)

    print('end_diff w ', end_diff_w)

    start_diff_v = np.round(start_diff_w / segm_spacing).astype(int)
    print('start diff v: ', start_diff_v)

    end_diff_v = np.round(end_diff_w / segm_spacing).astype(int)
    print('end diff v: ', end_diff_v)

    segm_end_v = img_dim_v - end_diff_v
    print('segm end v: ', segm_end_v)

    segm_end_x = segm_end_v[0]
    segm_end_y = segm_end_v[1]
    segm_end_z = segm_end_v[2]
    segm_start_x = segm_end_x - segm_dim_v[0]
    segm_start_y = segm_end_y - segm_dim_v[1]
    segm_start_z = segm_end_z - segm_dim_v[2]
    print('segm start v: ', segm_start_x, segm_start_y, segm_start_z)
    
    return segm_start_x, segm_start_y, segm_start_z, segm_end_x, segm_end_y, segm_end_z

def get_freequent_shape(arr, axis=0):
    arr = np.array(arr)
    u, indices = np.unique(arr, return_inverse=True)
    f_shape = u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(arr.shape),
                                    None, np.max(indices) + 1), axis=axis)]
    return f_shape

def makeit_3d(img):
    data = img.get_fdata()
    data = np.round(data)
    dims = len(img.shape)
    if dims == 3:
        data_3d = data
    elif dims == 4:
        data_3d = np.squeeze(data[:,:,:,0])
    else:
        raise Exception(f"Image with other than 3 or 4 dimention provided. Actual dim: {dims}")
    img_3d = nb.Nifti1Image(data_3d, img.affine, img.header)
    return img_3d

def volume_3_view_viewer(vol):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, tight_layout=True)
    axis = vol.shape

    ax1.imshow(vol[axis[0] // 2])
    ax2.imshow(vol[:, axis[1] // 2, :])
    ax3.imshow(vol[:, :, axis[2] // 2])

    plt.show()
    
def visualize_and_save(volid, vol_root=f'{processed_dir}/label_cropped', label_root=f'{processed_dir}/volume_cropped', img_save_path = f'{processed_dir}/merged_imgs'):
    vol = nb.load(f'{vol_root}/{volid}.nii.gz')
    label = nb.load(f'{label_root}/{volid}.nii.gz')

    im = vol.get_fdata()
    x = im.shape[1]//2
    masked = label.get_fdata()
    plt.figure()
    plt.imshow(im[:,x,:], 'gray', interpolation='none')
    plt.imshow(masked[:,x,:], 'jet', interpolation='none', alpha=0.5)
    plt.savefig(f'{img_save_path}/{volid}.png',  dpi=250, quality=95)
    plt.show()
    
def drop_overlapped_pixels(img, availed_manual_segs_id_list):
    labelmap = img.get_fdata()
    present_seg_idxs = np.unique(labelmap)
    overlapped_seg_idxs = set(present_seg_idxs).difference(availed_manual_segs_id_list)

    for idxs in overlapped_seg_idxs:
        print(f'Overlapped Idxs Found, removing it for idx {idxs}')
        labelmap[labelmap == idxs] = LABEL_EXTENSION_FOR_OVERLAP_REMOVAL
    
    labelmap -= LABEL_EXTENSION_FOR_OVERLAP_REMOVAL
    return nb.Nifti1Image(labelmap, img.affine, img.header)

def hist_match(img, histogram_matching_reference_path=HIST_MATCHING_VOL_PATH):
    if histogram_matching_reference_path is None:
        return img
    
    print(f"Initiating histogram match...")
    volume = img.get_fdata()
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

    hist_mapped_volume = interp_t_values[bin_idx].reshape(oldshape)
    return nb.Nifti1Image(hist_mapped_volume, img.affine, img.header)

def crop(paths, shape):
    s1, e1, s2, e2, s3, e3 = shape
    for path in paths:
        img = nb.load(path)
        img_data= img.get_fdata()
        data = img_data[s1:e1, s2:e2, s3:e3]
        img = nb.Nifti1Image(data, img.affine, img.header)
        save_path = '/'.join(path.split('/')[:-1])
        vol_id = path.split('/')[-1].split('.')[0]
        save_volume(img, f'{save_path}_cropped/{vol_id}')
        
class MRIDataset(data.Dataset):
    def __init__(self, X_files, y_files, transforms=None):
        self.X_files = X_files
        self.y_files = y_files
        self.transforms = transforms
    
        img_array = list()
        label_array = list()
        for vol_f, label_f in zip(self.X_files, self.y_files):
            img, label = nb.load(vol_f), nb.load(label_f)
            img_array.extend(np.array(img.get_fdata()))
            label_array.extend(np.array(label.get_fdata()))
            img.uncache()
            label.uncache()
            
        X = np.stack(img_array, axis=0) if len(img_array) > 1 else img_array[0]
        y = np.stack(label_array, axis=0) if len(label_array) > 1 else label_array[0]
        self.X = X if len(X.shape) == 4 else X[:, np.newaxis, :, :]
        self.y = y
        print(self.X.shape, self.y.shape)
        
    def __getitem__(self, index):
        img = torch.from_numpy(self.X[index])
        label = torch.from_numpy(self.y[index])
        return img, label

    def __len__(self):
        return len(self.y)
