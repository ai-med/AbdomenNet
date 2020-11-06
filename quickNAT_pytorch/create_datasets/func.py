import operator
import os
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
import nrrd
import glob
import nibabel

import SimpleITK as sitk
import sys

import pickle as p
import json
import subprocess
import getpass
import matplotlib.pyplot as plt

from nibabel.orientations import axcodes2ornt, ornt_transform, inv_ornt_aff, flip_axis
from nibabel.affines import from_matvec, to_matvec, apply_affine
from nibabel.processing import resample_to_output, resample_from_to
import numpy.linalg as npl

# from nilearn.image import resample_img
from global_vars import *

def SITK_N4_normalization(in_input_file, opp_file, output_file, shrink_factor=3,
                          mask=None, iteration=500, fittingLevel=4, tolerance=1e-03,
                          spline_param=200):
#     sitk.GetImageFromArray(nb.load(in_input_file).get_fdata())
# sitk.Cast
#     inputImage = sitk.Cast(sitk.GetImageFromArray(nb.load(in_input_file).get_fdata()), sitk.sitkFloat32) #sitk.ReadImage(in_input_file, sitk.sitkFloat32)
#     oppImage = sitk.Cast(sitk.GetImageFromArray(nb.load(opp_file).get_fdata()), sitk.sitkFloat32) #sitk.ReadImage(opp_file, sitk.sitkFloat32)
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
    
def nrrd_to_nifti(file_path):
    #output_file_path = os.path.join(file_path,'test')+'out.nii.gz'

    data, header = nrrd.read(file_path)

    if len(data.shape) == 4:
        data = data[:, :, :, 0]

    affine = header['space directions']
    affine = affine[:3, :3]
    origins = header['space origin']
    origins = origins[:3]
    t_mat = from_matvec(affine, origins)
    img = nibabel.Nifti1Image(data, t_mat)
    #nibabel.save(img, output_file_path)
    #img = nibabel.load(output_file_path)

    return img

# def reorient_image(image, axes='RAS', translation_params=(0, 0, 0), is_label=False):
#     # Global parameters
#     POSSIBLE_AXES_ORIENTATIONS = [
#         "LAI", "LIA", "ALI", "AIL", "ILA", "IAL",
#         "LAS", "LSA", "ALS", "ASL", "SLA", "SAL",
#         "LPI", "LIP", "PLI", "PIL", "ILP", "IPL",
#         "LPS", "LSP", "PLS", "PSL", "SLP", "SPL",
#         "RAI", "RIA", "ARI", "AIR", "IRA", "IAR",
#         "RAS", "RSA", "ARS", "ASR", "SRA", "SAR",
#         "RPI", "RIP", "PRI", "PIR", "IRP", "IPR",
#         "RPS", "RSP", "PRS", "PSR", "SRP", "SPR"
#     ]

#     #Load the image to rectify
# #     if in_file.split('.')[-1] == 'nrrd':
# #         print('NRRD file received, converting to nifti!!!')
# #         image = nrrd_to_nifti(in_file)
# #     else:
# #         image = nibabel.load(in_file)

#     header = image.header   
#     axes = nibabel.orientations.aff2axcodes(header.get_best_affine())
#     axes = "".join(list(axes))

#     # Check that a valid input axes is specified
#     if axes not in POSSIBLE_AXES_ORIENTATIONS:
#         raise ValueError("Wrong coordinate system: {0}.".format(axes))

#     rotation = swap_affine(axes)
#     det = np.linalg.det(rotation)
#     if det != 1:
#         raise Exception("Rotation matrix determinant must be one "
#                         "not '{0}'.".format(det))

#     affine = image.get_affine()
#     # Get the trnaslation to apply
#     translation = np.eye(4)
#     translation[:3, 3] = translation_params  # tuple(t) #if not is_label else translation_params[:3, 3]
#     transformation = np.dot(np.dot(rotation, affine), translation)
#     image.set_sform(transformation)
#     data = image.get_data()
#     header = image.header
#     new_axes = nibabel.orientations.aff2axcodes(header.get_best_affine())
#     new_axes = "".join(list(new_axes))
#     return image, data, header

def change_label(seg, f):
    print(f)
    seg_label = None
    if 'liver' in f or 'Liver' in f:
        print('found liver')
        seg[seg == 1] = 1
        seg_label = 'liver'
        
    elif 'spleen' in f or 'Spleen' in f:
        print('found spleen')
        seg[seg == 1] = 2
        seg_label = 'spleen'
    elif ('kidney' in f or 'Kidney' in f) and ('right' in f  or 'Right' in f):
        seg[seg==1] = 3
        print('found right kidney')
        seg_label = 'kidney_r'
        
    elif ('kidney' in f or 'Kidney' in f) and ('left' in f  or 'Left' in f): 
        seg[seg==1] = 4
        print('found left kidney')
        seg_label = 'kidney_l'

    
    elif ('adrenal' in f or 'Adrenal' in f) and ('right' in f  or 'Right' in f): 
        seg[seg==1] = 5
        print('found right adrenal')
        seg_label = 'adrenal_r'
        
    elif ('adrenal' in f or 'Adrenal' in f) and ('left' in f  or 'Left' in f):
        seg[seg==1] = 6
        print('found left adrenal')
        seg_label = 'adrenal_l'
        
    elif ('pancreas' in f or 'Pancreas' in f):
        seg[seg==1] = 7
        print('found pancreas')
        seg_label = 'pancreas'
    
    elif ('gallbladder' in f or 'Gallbladder' in f):
        seg[seg==1] = 8
        print('found gallbladder')
        seg_label = 'gallbladder'
    else:
        seg[seg==1] = 0
        print('found other organ, setting labels to 0')
        
    return seg, seg_label


def combine_nako_seg(seg_files, data_dir, vol_path):
    #data_dir = seg_dir
#     img_file = [f for f in seg_files if '3D_GRE_TRA_opp_3D_GRE_TRA_2.nii.gz' in f]
#     print('img file: ', img_file)
#     if len(img_file) == 1:
#         img_file = img_file[0]
#     else:
#         print('less or more than 1 img file found')

    print('data_dir ', data_dir)
#     print('img_file ', img_file)
    img = nibabel.load(vol_path)
    #print(img)
    img_h = img.header
    #print(img_h)
    print('x: ', img_h['qoffset_x'])
    print('y: ', img_h['qoffset_y'])
    print('z: ', img_h['qoffset_z'])

    # somehow this is wrong. x, and y should be positive and z negative

    segmentations = []
    for file in seg_files:
        if "nrrd" in file:
            seg, header = nrrd.read(file)

            # change labels according to label list above. liver stays the same, as it gets label=1         
            seg = change_label(seg, file)

            # print out some important header information:
            ##print(file)
            space_directions = header['space directions']
            space_origin = header['space origin']
            sizes = header['sizes']
            #print('z: ', space_origin[2])
            #print('size z: ', sizes[2])
            #print('spacing z: ', space_directions[2,2])

            segmentations.append((seg,header)) # append tuple of seg and header, need header later
    #print(segmentations)

    #print(len(segmentations))
    num_segm = len(segmentations)
    if num_segm < 8:
        print('less than 8 segmentations')

    segm_sorted = sorted(segmentations, key=lambda seg: seg[1]['space origin'][2], reverse=True)

    print('reorient image')
    img, data, img_h = reorient_image(img, is_label=False)
    
    comb_segm = np.zeros_like(img.get_data())
    print('final shape: ', comb_segm.shape)

    img_dim_v = img.shape


    for seg in segm_sorted:
        segm = seg[0]
        header = seg[1]
        segm_off = header['space origin']
        segm_dim_v = header['sizes']
        segm_spacing = abs(np.diag(header['space directions']))

        # if same size, just add the segmentation
        if all(segm_dim_v == img_dim_v):
            print('same')
            comb_segm += segm
        else:
            print('im dim v: ', img_dim_v)
            print('segm dim v: ', segm_dim_v)

            im_spacing = abs(img_h['pixdim'][1:4])
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
            
            label = segm[0]
            print(label.shape)
            print(comb_segm[segm_start_x:segm_end_x, segm_start_y:segm_end_y, segm_start_z:segm_end_z].shape)
            comb_segm[segm_start_x:segm_end_x, segm_start_y:segm_end_y, segm_start_z:segm_end_z] += label

    empty_header = nibabel.Nifti1Header()
    new_img_nii = nibabel.Nifti1Image(comb_segm, img.affine, empty_header)
    new_img_nii.header['pixdim'] = img_h['pixdim']

    #print(new_img_nii.header)
#     new_file_path = os.path.join(processed_dir,'combined_segmentation.nii.gz')
#     nibabel.save(new_img_nii, new_file_path)
    
    return img, new_img_nii

# def swap_affine(axes):
#     CORRECTION_MATRIX_COLUMNS = {
#         "R": (1, 0, 0),
#         "L": (-1, 0, 0),
#         "A": (0, 1, 0),
#         "P": (0, -1, 0),
#         "S": (0, 0, 1),
#         "I": (0, 0, -1)
#     }

#     if axes not in ['RSP', 'LIP', 'RAS', 'LPS']:
#         raise Exception(
#             f'Unknown axes passed for affine transformation! Please add transformation for {axes} manually.')
#     rotation = np.eye(4)
#     rotation[:3, 0] = CORRECTION_MATRIX_COLUMNS[axes[0]]
#     rotation[:3, 1] = CORRECTION_MATRIX_COLUMNS[axes[1]]
#     rotation[:3, 2] = CORRECTION_MATRIX_COLUMNS[axes[2]]
#     # print(rotation)
#     if axes == "RSP":
#         rotation = np.array([[1., 0., 0., 0.],
#                              [0., 0., 1., 0.],
#                              [0., -1., 0., 0.],
#                              [0., 0., 0., 1.]])
#         # print(rotation)
#     elif axes == "LPS":
#         rotation = np.array([
#             [-1., 0., 0., 0.],
#             [0., -1., 0., 0.],
#             [0., 0., 1., 0.],
#             [0., 0., 0., 1.]
#         ])
#         # print(rotation)
#     return rotation

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

def label_stitching(label_parts, img_reference):
    header = img_reference.header
    target_affine = img_reference.affine
    steps = header['pixdim'][1:4]
    volume = img_reference.get_fdata()
    im_dim_v = volume.shape
    
    processed_segm = None#np.zeros((500,500,1000))

    im_offx = abs(header['qoffset_x'])
    im_offy = abs(header['qoffset_y'])
    im_offz = abs(header['qoffset_z'])
    q_offsets = [im_offx,im_offy,im_offz]
    
    reference_labelmap = None
    for labelmap_img, lidx, lname in label_parts:
        print(lidx, lname)
        lheader = labelmap_img.header
        laffine = labelmap_img.affine
        if reference_labelmap is None:
            reference_labelmap = labelmap_img
        else:
            labelmap_img = resample_from_to(labelmap_img, [reference_labelmap.shape, reference_labelmap.affine])
            print('c shape:', labelmap_img.shape, labelmap_img.affine)
        
        sx,sy,sz, ex,ey,ez = get_points(labelmap_img, img_reference)
        
#         label_header = labelmap_img.header
        labelmap = labelmap_img.get_fdata()
#         labelmap_affine = labelmap_img.affine
#         steps_l = label_header['pixdim'][1:4]
#         lblmp_offx = abs(label_header['qoffset_x'])
#         lblmp_offy = abs(label_header['qoffset_y'])
#         lblmp_offz = abs(label_header['qoffset_z'])
#         q_offsets_l = [lblmp_offx,lblmp_offy,lblmp_offz]
        
#         print(f'vol offsets: {q_offsets}')
        
#         print(f'label_offsets: {q_offsets_l}')
        
# #         l_vol = np.prod(labelmap.shape)
#         segm_dim_v = labelmap.shape
        
#         ############
#         labelmap2vol = npl.inv(target_affine).dot(labelmap_affine)
#         seg_start_inv = np.floor(apply_affine(labelmap2vol, [0,0,0])).astype(np.int32)
#         seg_end_inv = apply_affine(labelmap2vol, segm_dim_v).astype(np.int32)
#         print("seg start inv v: ",seg_start_inv , "segm end inv v:",seg_end_inv)
#         ############
# #         shifts = tuple(np.floor(labelmap2vol[:3, 3]).astype(np.int32))
#         for ax, shift in enumerate(seg_start_inv):
#             print(ax, shift)
#             shift = int(shift)
#             if ax!=2:
# #                 labelmap = flip_axis(labelmap, axis=ax)
#                 if shift < 0:
#                     labelmap = flip_axis(labelmap, axis=ax)
#                     seg_end_inv[ax] += abs(seg_start_inv[ax])
#                     seg_start_inv[ax] = 0
#                 else:
# #                     labelmap = flip_axis(labelmap, axis=ax)
#                     seg_end_inv[ax] -= abs(seg_start_inv[ax])
#                     seg_start_inv[ax] = 0
            
#         print("updated seg start inv v: ",seg_start_inv , "updated segm end inv v:",seg_end_inv)
        labelmap = np.multiply(lidx, labelmap)
        
        if processed_segm is None:
#             seg_vol = np.prod(seg_end_inv)
#             vol_vol = np.prod(volume.shape)
#             processed_segm = np.empty(seg_end_inv) if seg_vol>vol_vol else np.zeros(volume.shape)
            processed_segm = labelmap
        else:
            processed_segm = np.add(processed_segm,labelmap)
        print("###############################################################################################")

#     print(volume.shape, processed_segm.shape)
#     processed_segm = np.flip(processed_segm)
    labelmap = np.round(processed_segm)
    empty_header = nb.Nifti1Header()
#     processed_segm_img = nb.Nifti1Image(processed_segm, target_affine, empty_header)
#     processed_segm = resample_from_to(processed_segm_img, [volume.shape, target_affine])
    
    
#     volume_ = np.zeros_like(processed_segm)
#     vx, vy, vz = volume.shape
#     volume_[0:vx, 0:vy, 0:vz] = volume
#     volume = volume_
#     volume = np.flip(np.moveaxis(volume, 2, 1))  # 012 -> 021
#     labelmap = np.flip(np.moveaxis(labelmap, 2, 1))
    volume = normalise_data(volume)

    
    stitched_labeled_img = nb.Nifti1Image(labelmap, laffine, lheader)
    volume_img = nb.Nifti1Image(volume, target_affine, header)
    
    return volume_img, stitched_labeled_img

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

# def labels_integerify(img):
#     data = img.get_fdata()
#     data = np.abs(np.round(data))
#     return nb.Nifti1Image(data, img.affine, img.header)

def label_parts(label_parts):
    stitched_label = None
    reference_labelmap = None
    mode = 'constant'
    label_shape = np.max([img.shape for img, _, _ in label_parts], axis=0)
    print('final_label_stitching shape:',label_shape)
    stitched_label = np.zeros(label_shape)
    for labelmap_img, lidx, lname in label_parts:
        print('lp:bfr:', lidx, lname, labelmap_img.shape, np.unique(labelmap_img.get_fdata()))
#         labelmap_img = makeit_3d(labelmap_img)
#         if reference_labelmap is None:
#             reference_labelmap = labelmap_img
#         else:
#             labelmap_img = resample_from_to(labelmap_img, [reference_labelmap.shape, reference_labelmap.affine], order=3, mode=mode, cval=0)
#             labelmap_img = labels_integerify(labelmap_img)
#         print(np.unique(labelmap_img.get_fdata()))
        
        labelmap = labelmap_img.get_fdata()
        labelmap = np.multiply(lidx, labelmap)
        x, y, z = labelmap.shape
        stitched_label[:x, :y, :z] += labelmap
#         if stitched_label is None:
#             stitched_label = labelmap
#         else:
#             print(stitched_label.shape)
#             stitched_label = np.add(stitched_label, labelmap)
        print("###############################################################################################") 
        
    labelmap = np.round(stitched_label)
    empty_header = nb.Nifti1Header()
    stitched_labeled_img = nb.Nifti1Image(labelmap, np.diag(TARGET_RESOLUTION+[1]))
    
    return stitched_labeled_img

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
    save_dir = '/'.join(file_name.split('/')[:-1])
    print("saving directory:", save_dir)
    create_if_not(save_dir)
    nb.save(img, f'{file_name}.nii.gz')


def sigmoid(x):
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = 1 / (1 + np.exp(-x[i]))
    return y


def normalise_data(volume):
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    return volume

# def post_interpolate(volume, labelmap=None, target_shape=[256,256,128]):
#     volume = do_cropping(volume, target_shape)
#     if labelmap is not None:
#         labelmap = do_cropping(labelmap, target_shape)
#     current_shape = volume.shape
#     intended_shape_deficit = target_shape - np.asarray(current_shape)

#     paddings = [tuple(
#         np.array([np.ceil((pad_tuples / 2) - pad_tuples % 2), np.floor((pad_tuples / 2) + pad_tuples % 2)]).astype(
#             'int32')) for pad_tuples in intended_shape_deficit]
#     paddings = tuple(paddings)

#     volume = np.pad(volume, paddings, mode='constant')
#     if labelmap is not None:
#         labelmap = np.pad(labelmap, paddings, mode='constant')

#     return volume, labelmap

# def do_cropping(source_num_arr, bounding):
#     start = list(map(lambda a, da: a // 2 - da // 2, source_num_arr.shape, bounding))
#     end = list(map(operator.add, start, bounding))
#     for i, val in enumerate(zip(start, end)):
#         if val[0] < 0:
#             start[i] = 0
#             end[i] = source_num_arr.shape[i]
#     slices = tuple(map(slice, tuple(start), tuple(end)))
#     return source_num_arr[slices]

def drop_overlapped_pixels(labelmap, availed_manual_segs_id_list, no_of_class):
    present_seg_idxs = np.unique(labelmap)
    overlapped_seg_idxs = set(present_seg_idxs).difference(availed_manual_segs_id_list)

    for idxs in overlapped_seg_idxs:
        print(f'Overlapped Idxs Found, removing it for idx {idxs}')
        labelmap[labelmap == idxs] = 0
    return labelmap

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
    elif file_type == 'gz' or file_type == 'nii':
        data, header, img = nibabel_reader(file_path)
        header_mat = header
    else:
        print(f'Unknown file type: {file_type} for file path: {file_path}')
        data, header_mat, img = None, None, None

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
        img_1 = resample_to_output(img_1, TARGET_RESOLUTION, order=3, mode=mode, cval=0.0)
        target_affine = img_0.affine.copy()
        target_affine[2, 3] = img_1.affine[2, 3].copy()
        target_shape = img_0.shape[:2] + img_1.shape[2:]
        img_1 = resample_from_to(img_1, [target_shape, target_affine])
        img_0 = vol_stitching(img_0, img_1)
    return img_0

def visualize_and_save(volid, 
              vol_root=f'{processed_dir}/label_cropped', 
              label_root=f'{processed_dir}/volume_cropped',
             img_save_path = f'{processed_dir}/merged_imgs'):
    vol = nb.load(f'{vol_root}/{volid}.nii.gz')
    label = nb.load(f'{label_root}/{volid}.nii.gz')

    im = vol.get_fdata()
    x = im.shape[0]//2
    masked = label.get_fdata()
    plt.figure()
#     plt.subplot(1,2,1)
#     plt.imshow(im[x], 'gray', interpolation='none')
#     plt.subplot(1,2,2)
    plt.imshow(im[x], 'jet', interpolation='none')
    plt.imshow(masked[x], 'gray', interpolation='none', alpha=0.5)
    plt.savefig(f'{img_save_path}/{volid}.png',  dpi=250, quality=95)
    plt.show()
    

def visualize_overlay(file_paths):
    for vol_id in file_paths.keys():
        print(vol_id)
        try:
            vol = nb.load(f'{processed_dir}/volume/{vol_id}.nii.gz')
            label = nb.load(f'{processed_dir}/label/{vol_id}.nii.gz')
        except Exception as e:
            print(e)
            continue
        im = vol.get_fdata()
        x = im.shape[0]//2
        masked = label.get_fdata()
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(im[x], 'gray', interpolation='none')
        plt.subplot(1,2,2)
        plt.imshow(im[x], 'gray', interpolation='none')
        plt.imshow(masked[x], 'jet', interpolation='none', alpha=0.5)
        plt.show()

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

# def placing_axes(vol, target_affine, target_header=None, skip_axis=None):
#     vol2target = npl.inv(target_affine).dot(vol.affine)
#     source_data = vol.get_fdata()
#     shifts = tuple(vol2target[:3, 3].astype(np.int32))
# #     print(shifts)
#     #     print(source_data.shape)
#     for ax, shift in enumerate(shifts):
#         if skip_axis is not None and ax in skip_axis:
#             continue
#         print(ax, shift)
#         shift = int(shift)
#         if shift < 0:
#             source_data = flip_axis(source_data, axis=ax)
#         print(-np.abs(shift))
#         source_data = np.roll(source_data, -np.abs(shift), axis=ax)

#     if target_header is None:
#         target_header = nb.Nifti1Header()
#     stitched_labeled_img = nb.Nifti1Image(source_data, target_affine, target_header)

#     return stitched_labeled_img



def get_freequent_shape(arr, axis=0):
    arr = np.array(arr)
    u, indices = np.unique(arr, return_inverse=True)
    f_shape = u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(arr.shape),
                                    None, np.max(indices) + 1), axis=axis)]
    return f_shape

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



# def label_stitch_extd(images, is_label=False):
#     if len(images) == 1:
#         return images
#     elif len(images) == 0:
#         raise Exception("Empty Image List!")

# #     images_sorted = sorted(images, key=lambda im: im.header['qoffset_z'], reverse=True)
# #     img_0 = images_sorted[0]

# #     mode = 'nearest' if is_label else 'constant'
# #     img_0 = resample_to_output(img_0, TARGET_RESOLUTION, order=3, mode=mode, cval=0.0)
    
#     processed_segm = None #np.zeros_like(img_0.get_data())
#     reference_labelmap = None
#     target_affine = images[0][0].affine
#     mode = 'nearest'
#     for im_1, lidx, labelname in images:
#         print(im_1.shape, lidx, labelname)
#         if reference_labelmap is None:
#             reference_labelmap = im_1
#         else:
#             im_1 = resample_from_to(im_1, [reference_labelmap.shape, reference_labelmap.affine], mode=mode)
            
#         print(im_1.shape, lidx, labelname)    
    
#         im_1_x, im_1_y, im_1_z = im_1.shape
        
#         im_1_start_width_x = abs(im_1.header['qoffset_x'])
#         im_1_start_width_y = abs(im_1.header['qoffset_y'])
#         im_1_start_width_z = abs(im_1.header['qoffset_z'])

#         spacing_img_1_x, spacing_img_1_y, spacing_img_1_z = im_1.header['pixdim'][1:4]
#         print(spacing_img_1_x, spacing_img_1_y, spacing_img_1_z)

#         im_1_width_x = im_1_x * spacing_img_1_x
#         im_1_width_y = im_1_y * spacing_img_1_y
#         im_1_width_z = im_1_z * spacing_img_1_z

#         im_1_end_width_x = im_1_start_width_x + im_1_width_x
#         im_1_end_width_y = im_1_start_width_y + im_1_width_y
#         im_1_end_width_z = im_1_start_width_z + im_1_width_z
        
#         im_1_end_x = im_1_end_width_x // spacing_img_1_x
#         im_1_end_y = im_1_end_width_y // spacing_img_1_y
#         im_1_end_z = im_1_end_width_z // spacing_img_1_z
        
#         im_1_start_x = im_1_start_width_x // spacing_img_1_x
#         im_1_start_y = im_1_start_width_y // spacing_img_1_y
#         im_1_start_z = im_1_start_width_z // spacing_img_1_z

#         im_1_data = im_1.get_fdata()
        
#         im_1_data = np.multiply(lidx, im_1_data)
        
#         im_1_start_x,im_1_end_x, im_1_start_y,im_1_end_y, im_1_start_z,im_1_end_z = int(im_1_start_x),int(im_1_end_x), int(im_1_start_y),int(im_1_end_y), int(im_1_start_z),int(im_1_end_z)
#         print(im_1_start_x,im_1_end_x, im_1_start_y,im_1_end_y, im_1_start_z,im_1_end_z)
#         if processed_segm is None:
#             processed_segm = np.zeros((im_1_end_x, im_1_end_y, im_1_end_z))
# #             print(processed_segm.shape)
        
#         processed_segm[int(im_1_start_x):int(im_1_end_x), int(im_1_start_y):int(im_1_end_y), int(im_1_start_z):int(im_1_end_z)] += im_1_data
    
#     labelmap = np.round(processed_segm)
    
#     empty_header = nb.Nifti1Header()
#     s_labelmap = nb.Nifti1Image(labelmap, target_affine, empty_header)
    
#     return None, s_labelmap



