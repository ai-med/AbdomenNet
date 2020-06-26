import os

import h5py
import nibabel as nb
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
import utils.preprocessor as preprocessor
#import preprocessor
# transform_train = transforms.Compose([
#     transforms.RandomCrop(200, padding=56),
#     transforms.ToTensor(),
# ])


class ImdbData(data.Dataset):
    def __init__(self, X, y, w, wd, transforms=None):
        self.X = X if len(X.shape) == 4 else X[:, np.newaxis, :, :]
        self.y = y
        self.w = w
        self.wd = wd
        self.transforms = transforms

    def __getitem__(self, index):
        img = torch.from_numpy(self.X[index])
        label = torch.from_numpy(self.y[index])
        weight = torch.from_numpy(self.w[index])
        dice_weight = torch.from_numpy(self.wd[index])
        return img, label, weight, dice_weight

    def __len__(self):
        return len(self.y)

class ImdbData_3channel(data.Dataset):
    def __init__(self, X, X2, X3, y, w, wd, transforms=None):
        self.X = X if len(X.shape) == 4 else X[:, np.newaxis, :, :]
        self.X2 = X2 if len(X2.shape) == 4 else X2[:, np.newaxis, :, :]
        self.X3 = X3 if len(X3.shape) == 4 else X3[:, np.newaxis, :, :]
        self.y = y
        self.w = w
        self.wd = wd
        self.transforms = transforms

    def __getitem__(self, index):
        img1 = torch.from_numpy(self.X[index])
        img2 = torch.from_numpy(self.X2[index])
        img3 = torch.from_numpy(self.X3[index])
        img = torch.cat([img1,img2,img3], dim=0)
        label = torch.from_numpy(self.y[index])
        weight = torch.from_numpy(self.w[index])
        dice_weight = torch.from_numpy(self.wd[index])
        return img, label, weight, dice_weight

    def __len__(self):
        return len(self.y)

def get_imdb_dataset_3channel(data_params):
    data_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_data_file']), 'r')
    fat_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_data_fat_file']), 'r')
    water_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_data_water_file']), 'r')
    label_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_label_file']), 'r')
    class_weight_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_class_weights_file']), 'r')
    weight_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_weights_file']), 'r')

    data_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_data_file']), 'r')
    fat_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_data_fat_file']), 'r')
    water_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_data_water_file']), 'r')
    label_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_label_file']), 'r')
    class_weight_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_class_weights_file']), 'r')
    weight_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_weights_file']), 'r')

    return (ImdbData_3channel(data_train['data'][()],fat_train['data'][()],water_train['data'][()], label_train['label'][()], class_weight_train['class_weights'][()], weight_train['weights'][()]),
            ImdbData_3channel(data_test['data'][()],fat_test['data'][()],water_test['data'][()], label_test['label'][()], class_weight_test['class_weights'][()], weight_test['weights'][()]))

def get_imdb_dataset(data_params):
    data_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_data_file']), 'r')
    label_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_label_file']), 'r')
    class_weight_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_class_weights_file']), 'r')
    weight_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_weights_file']), 'r')

    data_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_data_file']), 'r')
    label_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_label_file']), 'r')
    class_weight_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_class_weights_file']), 'r')
    weight_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_weights_file']), 'r')

    return (ImdbData(data_train['data'][()], label_train['label'][()], class_weight_train['class_weights'][()], weight_train['weights'][()]),
            ImdbData(data_test['data'][()], label_test['label'][()], class_weight_test['class_weights'][()], weight_test['weights'][()]))


def load_dataset_3channel(file_paths,
                 orientation,
                 remap_config,
                 return_weights=False,
                 reduce_slices=False,
                 remove_black=False):
    print("Loading and preprocessing data...")
    volume_list, fat_list, water_list, labelmap_list, headers, class_weights_list, weights_list = [], [], [], [], [], [],[]

    for file_path in file_paths:
        volume, fat,water, labelmap, class_weights, weights, header = load_and_preprocess_3channel(file_path, orientation,
                                                                               remap_config=remap_config,
                                                                               reduce_slices=reduce_slices,
                                                                               remove_black=remove_black,
                                                                               return_weights=return_weights)

        volume_list.append(volume)
        fat_list.append(fat)
        water_list.append(water)
        labelmap_list.append(labelmap)

        if return_weights:
            class_weights_list.append(class_weights)
            weights_list.append(weights)

        headers.append(header)

        print("#", end='', flush=True)
    print("100%", flush=True)
    if return_weights:
        return volume_list, fat_list, water_list ,labelmap_list, class_weights_list, weights_list, headers
    else:
        return volume_list,fat_list, water_list, labelmap_list, headers

def load_dataset(file_paths,
                 orientation,
                 remap_config,
                 return_weights=False,
                 reduce_slices=False,
                 remove_black=False):
    print("Loading and preprocessing data...")
    volume_list, labelmap_list, headers, class_weights_list, weights_list = [], [], [], [], []

    for file_path in file_paths:
        volume, labelmap, class_weights, weights, header = load_and_preprocess(file_path, orientation,
                                                                               remap_config=remap_config,
                                                                               reduce_slices=reduce_slices,
                                                                               remove_black=remove_black,
                                                                               return_weights=return_weights)

        volume_list.append(volume)
        labelmap_list.append(labelmap)

        if return_weights:
            class_weights_list.append(class_weights)
            weights_list.append(weights)

        headers.append(header)

        print("#", end='', flush=True)
    print("100%", flush=True)
    if return_weights:
        return volume_list, labelmap_list, class_weights_list, weights_list, headers
    else:
        return volume_list, labelmap_list, headers

def load_and_preprocess_3channel(file_path, orientation, remap_config, reduce_slices=False,
                        remove_black=False,
                        return_weights=False):
    #print(file_path)
    volume,fat, water, labelmap, header = load_data_3channel(file_path, orientation)

    volume, fat, water, labelmap, class_weights, weights = preprocess_3channel(volume, fat, water,labelmap, remap_config=remap_config,
                                                          reduce_slices=reduce_slices,
                                                          remove_black=remove_black,
                                                          return_weights=return_weights)
    return volume,fat,water, labelmap, class_weights, weights, header

def load_and_preprocess(file_path, orientation, remap_config, reduce_slices=False,
                        remove_black=False,
                        return_weights=False):
    #print(file_path)
    volume, labelmap, header = load_data(file_path, orientation)

    volume, labelmap, class_weights, weights = preprocess_3channel(volume,  labelmap, remap_config=remap_config,
                                                          reduce_slices=reduce_slices,
                                                          remove_black=remove_black,
                                                          return_weights=return_weights)
    return volume, labelmap, class_weights, weights, header


def load_and_preprocess_eval(file_path, orientation, notlabel=True):
    volume_nifty = nb.load(file_path[0])
    label_nifty = nb.load(file_path[1])

    header = volume_nifty.header
    volume = volume_nifty.get_fdata()
    label = label_nifty.get_data()
    label = label.astype('int')
    if notlabel:
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    else:
        volume = np.round(volume)
    if orientation == "AXI":
        volume = volume.transpose((2, 1,0))
    elif orientation == "COR":
        volume = volume.transpose((1,0,2))
    return volume, label, header


def load_data_3channel(file_path, orientation):
    volume_nifty, volume_fat_nifty, volume_water_nifty, labelmap_nifty = nb.load(file_path[0]),nb.load(file_path[1]),nb.load(file_path[2]), nb.load(file_path[3])
    volume, fat,water, labelmap = volume_nifty.get_fdata(),volume_fat_nifty.get_fdata(),volume_water_nifty.get_fdata(), labelmap_nifty.get_fdata()
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    fat = (fat - np.min(fat)) / (np.max(fat) - np.min(fat))
    water = (water - np.min(water)) / (np.max(water) - np.min(water))

    volume = preprocessor.rotate_orientation(volume, orientation)
    fat = preprocessor.rotate_orientation(fat, orientation)
    water = preprocessor.rotate_orientation(water, orientation)
    labelmap = preprocessor.rotate_orientation(labelmap, orientation)
    return volume, fat, water, labelmap, volume_nifty.header

def load_data(file_path, orientation):
    volume_nifty, labelmap_nifty = nb.load(file_path[0]), nb.load(file_path[1])
    volume, labelmap = volume_nifty.get_fdata(), labelmap_nifty.get_fdata()
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    labelmap = preprocessor.rotate_orientation(labelmap, orientation)
    volume = preprocessor.rotate_orientation(volume, orientation)
    return volume, labelmap, volume_nifty.header

def preprocess_3channel(volume, fat, water, labelmap, remap_config, reduce_slices=False, remove_black=False, return_weights=False):
    if reduce_slices:
        volume,fat,water, labelmap = preprocessor.reduce_slices_3channel(volume, labelmap)

    #if remap_config:
    #    labelmap = preprocessor.remap_labels(labelmap, remap_config)

    if remove_black:
        volume, fat,water, labelmap = preprocessor.remove_black_3channels(volume, labelmap)

    if return_weights:
        class_weights, weights = preprocessor.estimate_weights_mfb(labelmap)
        return volume, fat, water, labelmap, class_weights, weights
    else:
        return volume, fat, water, labelmap, None, None

def preprocess(volume, labelmap, remap_config, reduce_slices=False, remove_black=False, return_weights=False):
    if reduce_slices:
        volume, labelmap = preprocessor.reduce_slices(volume, labelmap)

    #if remap_config:
    #    labelmap = preprocessor.remap_labels(labelmap, remap_config)

    if remove_black:
        volume, labelmap = preprocessor.remove_black(volume, labelmap)

    if return_weights:
        class_weights, weights = preprocessor.estimate_weights_mfb(labelmap)
        return volume, labelmap, class_weights, weights
    else:
        return volume, labelmap, None, None


# def load_file_paths(data_dir, label_dir, volumes_txt_file=None):
#     """
#     This function returns the file paths combined as a list where each element is a 2 element tuple, 0th being data and 1st being label.
#     It should be modified to suit the need of the project
#     :param data_dir: Directory which contains the data files
#     :param label_dir: Directory which contains the label files
#     :param volumes_txt_file: (Optional) Path to the a csv file, when provided only these data points will be read
#     :return: list of file paths as string
#     """
#
#     volume_exclude_list = ['IXI290', 'IXI423']
#     if volumes_txt_file:
#         with open(volumes_txt_file) as file_handle:
#             volumes_to_use = file_handle.read().splitlines()
#     else:
#         volumes_to_use = [name for name in os.listdir(data_dir) if
#                           name.startswith('IXI') and name not in volume_exclude_list]
#
#     file_paths = [
#         [os.path.join(data_dir, vol, 'mri/orig.mgz'), os.path.join(label_dir, vol, 'mri/aseg.auto_noCCseg.mgz')]
#         for
#         vol in volumes_to_use]
#     return file_paths

def load_file_paths_3channel(data_dir, label_dir, data_id, volumes_txt_file=None):
    """
    This function returns the file paths combined as a list where each element is a 2 element tuple, 0th being data and 1st being label.
    It should be modified to suit the need of the project
    :param data_dir: Directory which contains the data files
    :param label_dir: Directory which contains the label files
    :param data_id: A flag indicates the name of Dataset for proper file reading
    :param volumes_txt_file: (Optional) Path to the a csv file, when provided only these data points will be read
    :return: list of file paths as string
    """

    if volumes_txt_file:
        with open(volumes_txt_file) as file_handle:
            volumes_to_use = file_handle.read().splitlines()
    else:
        volumes_to_use = [name for name in os.listdir(data_dir)]
    print('data id ', data_id)
    if data_id == 'KORA':
        file_paths = [
            [os.path.join(data_dir, data_id, 'data', vol, 'resampled_normalized_image.nii.gz'), os.path.join(data_dir, data_id, 'data', vol, 'resampled_normalized_fat.nii.gz'), os.path.join(data_dir, data_id, 'data', vol, 'resampled_normalized_water.nii.gz'), os.path.join(label_dir, data_id,'data',vol, 'resampled_segm.nii.gz')]
            for
            vol in volumes_to_use]
    elif data_id == 'NAKO':
        file_paths = [
            [os.path.join(data_dir, data_id,'data',vol, 'resampled_normalized_image.nii.gz'),
             os.path.join(label_dir, data_id,'data',vol, 'resampled_segm.nii.gz')]
            for
            vol in volumes_to_use]
    elif data_id == 'UKB':
        file_paths = [
            [os.path.join(data_dir, data_id,'data',vol, 'resampled_normalized_image.nii.gz'),
             os.path.join(label_dir, data_id,'data',vol, 'resampled_segm.nii.gz')]
            for
            vol in volumes_to_use]
    else:
        raise ValueError("Invalid entry, valid options are MALC, ADNI, CANDI and IBSR")

    return file_paths

def load_file_paths(data_dir, label_dir, data_id, volumes_txt_file=None):
    """
    This function returns the file paths combined as a list where each element is a 2 element tuple, 0th being data and 1st being label.
    It should be modified to suit the need of the project
    :param data_dir: Directory which contains the data files
    :param label_dir: Directory which contains the label files
    :param data_id: A flag indicates the name of Dataset for proper file reading
    :param volumes_txt_file: (Optional) Path to the a csv file, when provided only these data points will be read
    :return: list of file paths as string
    """

    if volumes_txt_file:
        with open(volumes_txt_file) as file_handle:
            volumes_to_use = file_handle.read().splitlines()
    else:
        volumes_to_use = [name for name in os.listdir(data_dir)]
    print('data id ', data_id)
    if data_id == 'KORA':
        file_paths = [
            [os.path.join(data_dir, data_id, 'data', vol, 'transformed_corrected.nii.gz'), os.path.join(label_dir, data_id,'data',vol, 'resampled_segm.nii.gz')]
            for
            vol in volumes_to_use]
    elif data_id == 'NAKO':
        file_paths = [
            [os.path.join(data_dir, data_id,'data',vol, 'resampled_normalized_image.nii.gz'),
             os.path.join(label_dir, data_id,'data',vol, 'resampled_segm.nii.gz')]
            for
            vol in volumes_to_use]
    elif data_id == 'UKB':
        file_paths = [
            [os.path.join(data_dir, data_id,'data',vol, 'resampled_normalized_image.nii.gz'),
             os.path.join(label_dir, data_id,'data',vol, 'resampled_segm.nii.gz')]
            for
            vol in volumes_to_use]
    else:
        raise ValueError("Invalid entry, valid options are MALC, ADNI, CANDI and IBSR")

    return file_paths


def load_file_paths_eval(data_dir, volumes_txt_file, dir_struct):
    """
    This function returns the file paths combined as a list where each element is a 2 element tuple, 0th being data and 1st being label.
    It should be modified to suit the need of the project
    :param data_dir: Directory which contains the data files
    :param volumes_txt_file:  Path to the a csv file, when provided only these data points will be read
    :param dir_struct: If the id_list is in FreeSurfer style or normal
    :return: list of file paths as string
    """

    with open(volumes_txt_file) as file_handle:
        volumes_to_use = file_handle.read().splitlines()
    if dir_struct == "FS":
        file_paths = [
            [os.path.join(data_dir, vol, 'mri/orig.mgz')]
            for
            vol in volumes_to_use]
    elif dir_struct == "Linear":
        file_paths = [
            [os.path.join(data_dir, vol)]
            for
            vol in volumes_to_use]
    elif dir_struct == "part_FS":
        file_paths = [
            [os.path.join(data_dir, vol, 'orig.mgz')]
            for
            vol in volumes_to_use]
    elif dir_struct == 'whole_body':
        file_paths = [
            [os.path.join(data_dir, vol, 'resampled_normalized_image.nii.gz')
             ]
            for
            vol in volumes_to_use

        ]
    else:
        raise ValueError("Invalid entry, valid options are FS and Linear")
    return file_paths
