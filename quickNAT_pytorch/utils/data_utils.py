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

class ImdbData_2channel(data.Dataset):
    def __init__(self, X, X2, y, w, wd, transforms=None):
        self.X = X if len(X.shape) == 4 else X[:, np.newaxis, :, :]
        self.X2 = X2 if len(X2.shape) == 4 else X2[:, np.newaxis, :, :]
        # self.X3 = X3 if len(X3.shape) == 4 else X3[:, np.newaxis, :, :]
        self.y = y
        self.w = w
        self.wd = wd
        self.transforms = transforms


    def __getitem__(self, index):
        img1 = torch.from_numpy(self.X[index])
        img2 = torch.from_numpy(self.X2[index])
        # img3 = torch.from_numpy(self.X3[index])
        img = torch.cat([img1,img2], dim=0)
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


def get_imdb_dataset_2channel(data_params):
    data_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_data_file']), 'r')
    # fat_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_data_fat_file']), 'r')
    water_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_data_water_file']), 'r')
    label_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_label_file']), 'r')
    class_weight_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_class_weights_file']), 'r')
    weight_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_weights_file']), 'r')

    data_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_data_file']), 'r')
    # fat_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_data_fat_file']), 'r')
    water_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_data_water_file']), 'r')
    label_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_label_file']), 'r')
    class_weight_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_class_weights_file']), 'r')
    weight_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_weights_file']), 'r')

    return (ImdbData_2channel(data_train['data'][()],water_train['data'][()], label_train['label'][()], class_weight_train['class_weights'][()], weight_train['weights'][()]),
            ImdbData_2channel(data_test['data'][()],water_test['data'][()], label_test['label'][()], class_weight_test['class_weights'][()], weight_test['weights'][()]))


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


def load_and_preprocess_2channel(file_path, orientation, remap_config, reduce_slices=False,
                        remove_black=False,
                        return_weights=False):
    #print(file_path)
    volume, water, labelmap, header = load_data_2channel(file_path, orientation)

    volume, water, labelmap, class_weights, weights = preprocess_2channel(volume, water,labelmap, remap_config=remap_config,
                                                          reduce_slices=reduce_slices,
                                                          remove_black=remove_black,
                                                          return_weights=return_weights)
    return volume, water, labelmap, class_weights, weights, header


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

    volume, labelmap, class_weights, weights = preprocess(volume,  labelmap, remap_config=remap_config,
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


def load_data_2channel(file_path, orientation):
    volume_nifty, volume_water_nifty, labelmap_nifty = nb.load(file_path[0]),nb.load(file_path[2]), nb.load(file_path[1])
    volume, water, labelmap = volume_nifty.get_fdata(),volume_water_nifty.get_fdata(), labelmap_nifty.get_fdata()
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    # fat = (fat - np.min(fat)) / (np.max(fat) - np.min(fat))
    water = (water - np.min(water)) / (np.max(water) - np.min(water))

    volume = preprocessor.rotate_orientation(volume, orientation)
    # fat = preprocessor.rotate_orientation(fat, orientation)
    water = preprocessor.rotate_orientation(water, orientation)
    labelmap = preprocessor.rotate_orientation(labelmap, orientation)
    return volume, water, labelmap, volume_nifty.header

def load_data(file_path, orientation):
    volume_nifty, labelmap_nifty = nb.load(file_path[0]), nb.load(file_path[1])
    volume, labelmap = volume_nifty.get_fdata(), labelmap_nifty.get_fdata()
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    labelmap = preprocessor.rotate_orientation(labelmap, orientation)
    volume = preprocessor.rotate_orientation(volume, orientation)
    return volume, labelmap, volume_nifty.header

def preprocess_3channel(volume, fat, water, labelmap, remap_config, reduce_slices=False, remove_black=False, return_weights=False):
    if reduce_slices:
        volume,fat,water, labelmap = preprocessor.reduce_slices_3channel(volume, fat, water, labelmap)

    #if remap_config:
    #    labelmap = preprocessor.remap_labels(labelmap, remap_config)

    # if remove_black:
    #     volume, fat,water, labelmap = preprocessor.remove_black_3channels(volume, labelmap)

    if return_weights:
        class_weights, weights = preprocessor.estimate_weights_mfb(labelmap)
        return volume, fat, water, labelmap, class_weights, weights
    else:
        return volume, fat, water, labelmap, None, None


def preprocess_2channel(volume, water, labelmap, remap_config, reduce_slices=False, remove_black=False, return_weights=False):
    if reduce_slices:
        volume, water, labelmap = preprocessor.reduce_slices_2channel(volume, water, labelmap)

    #if remap_config:
    #    labelmap = preprocessor.remap_labels(labelmap, remap_config)

    # if remove_black:
    #     volume, fat,water, labelmap = preprocessor.remove_black_3channels(volume, labelmap)

    if return_weights:
        class_weights, weights = preprocessor.estimate_weights_mfb(labelmap)
        return volume, water, labelmap, class_weights, weights
    else:
        return volume, water, labelmap, None, None


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

    file_paths = [
        [os.path.join(data_dir, f'{vol}.nii.gz'), os.path.join(label_dir, f'{vol}.nii.gz'), os.path.join(f'{data_dir}_w', f'{vol}.nii.gz'), os.path.join(f'{data_dir}_f', f'{vol}.nii.gz'), os.path.join(f'{data_dir}_in', f'{vol}.nii.gz')]
        for
        vol in volumes_to_use]

    return file_paths

    # print('data id ', data_id)
    # if data_id == 'KORA':
    #     file_paths = [
    #         [os.path.join(data_dir, data_id, 'data', vol, 'resampled_normalized_image.nii.gz'), os.path.join(data_dir, data_id, 'data', vol, 'resampled_normalized_fat.nii.gz'), os.path.join(data_dir, data_id, 'data', vol, 'resampled_normalized_water.nii.gz'), os.path.join(label_dir, data_id,'data',vol, 'resampled_segm.nii.gz')]
    #         for
    #         vol in volumes_to_use]
    # elif data_id == 'NAKO':
    #     file_paths = [
    #         [os.path.join(data_dir, data_id,'data',vol, 'resampled_normalized_image.nii.gz'),
    #          os.path.join(label_dir, data_id,'data',vol, 'resampled_segm.nii.gz')]
    #         for
    #         vol in volumes_to_use]
    # elif data_id == 'UKB':
    #     file_paths = [
    #         [os.path.join(data_dir, data_id,'data',vol, 'resampled_normalized_image.nii.gz'),
    #          os.path.join(label_dir, data_id,'data',vol, 'resampled_segm.nii.gz')]
    #         for
    #         vol in volumes_to_use]
    # else:
    #     raise ValueError("Invalid entry, valid options are MALC, ADNI, CANDI and IBSR")

    # return file_paths

def load_file_paths_2channel(data_dir, label_dir, data_id, volumes_txt_file=None):
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

    file_paths = [
        [os.path.join(data_dir, f'{vol}.nii.gz'), os.path.join(label_dir, f'{vol}.nii.gz'), os.path.join(f'{data_dir}_w', f'{vol}.nii.gz')]
        for
        vol in volumes_to_use]

    return file_paths

    # print('data id ', data_id)
    # if data_id == 'KORA':
    #     file_paths = [
    #         [os.path.join(data_dir, data_id, 'data', vol, 'resampled_normalized_image.nii.gz'), os.path.join(data_dir, data_id, 'data', vol, 'resampled_normalized_water.nii.gz'), os.path.join(label_dir, data_id,'data',vol, 'resampled_segm.nii.gz')]
    #         for
    #         vol in volumes_to_use]
    # elif data_id == 'NAKO':
    #     file_paths = [
    #         [os.path.join(data_dir, data_id,'data',vol, 'resampled_normalized_image.nii.gz'),
    #          os.path.join(label_dir, data_id,'data',vol, 'resampled_segm.nii.gz')]
    #         for
    #         vol in volumes_to_use]
    # elif data_id == 'UKB':
    #     file_paths = [
    #         [os.path.join(data_dir, data_id,'data',vol, 'resampled_normalized_image.nii.gz'),
    #          os.path.join(label_dir, data_id,'data',vol, 'resampled_segm.nii.gz')]
    #         for
    #         vol in volumes_to_use]
    # else:
    #     raise ValueError("Invalid entry, valid options are MALC, ADNI, CANDI and IBSR")

    # return file_paths


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
            [os.path.join(data_dir, f'{vol}.nii.gz'), os.path.join(label_dir, f'{vol}.nii.gz')]
            for
            vol in volumes_to_use]
    elif data_id == 'NAKO':
        file_paths = [
            [os.path.join(data_dir, f'{vol}.nii.gz'), os.path.join(label_dir, f'{vol}.nii.gz')]
            for
            vol in volumes_to_use]
        # file_paths = [
        #     [os.path.join(data_dir, data_id,'data',vol, 'resampled_normalized_image.nii.gz'),
        #      os.path.join(label_dir, data_id,'data',vol, 'resampled_segm.nii.gz')]
        #     for
        #     vol in volumes_to_use]
    elif data_id == 'UKB':
        file_paths = [
            [os.path.join(data_dir, f'{vol}.nii.gz'), os.path.join(label_dir, f'{vol}.nii.gz')]
            for
            vol in volumes_to_use]
        # file_paths = [
        #     [os.path.join(data_dir, data_id,'data',vol, 'resampled_normalized_image.nii.gz'),
        #      os.path.join(label_dir, data_id,'data',vol, 'resampled_segm.nii.gz')]
        #     for
        #     vol in volumes_to_use]
    else:
        raise ValueError("Invalid entry, valid options are MALC, ADNI, CANDI and IBSR")

    return file_paths


def _load_file_paths_(data_dir, label_dir, data_id, volumes_txt_file=None):
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

    if data_id == 'KORA':
        file_paths = [
            [os.path.join(data_dir, f'{vol}.nii.gz'),
             os.path.join(label_dir, f'{vol}.nii.gz')]
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

def estimate_weights_mfb(labels, no_of_class=9):
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

class MRIDataset(data.Dataset):

    def __init__(self, X_files, y_files, transforms=None, thickSlice=None, water_vols=None, fat_vols=None, in_vols=None, orientation='AXI', is_train=False, fold=False):
        # if fold:
        #     self.X = X_files
        #     self.y = y_files
        #     self.transforms = transform
        print(f'+++ Preparing data for {fold} fold +++')
        self.X_files = X_files
        self.y_files = y_files
        self.transforms = transforms
        self.thickSlice = thickSlice
        self.water_vols = water_vols
        self.fat_vols = fat_vols
        self.in_vols = in_vols

        self.is_train = is_train

        if orientation == 'AXI':
            self.to_axis = 2
        elif orientation == 'COR':
            self.to_axis = 1
        else:
            self.to_axis = 0

        # assert(self.thickSlice is None if self.water_vols is not None)
        # assert(self.water_vols is None if self.thickSlice is not None)

        img_array = list()
        label_array = list()
        water_array = list()
        # fat_array = list()
        in_array = list()
        cw_array = list()
        w_array = list()

        for vol_f, label_f, water_f in zip(self.X_files, self.y_files, self.water_vols):
        # for vol_f, label_f, water_f in zip(self.X_files, self.y_files, self.water_vols):
        # for vol_f, label_f in zip(self.X_files, self.y_files):
            img, label = nb.load(vol_f), nb.load(label_f)
            water = nb.load(water_f)
            # fat = nb.load(fat_f)
            # inv = nb.load(in_f)

            img_data = np.array(img.get_fdata())
            label_data = np.array(label.get_fdata())
            water_data = np.array(water.get_fdata())
            # fat_data = np.array(fat.get_fdata())
            # in_data = np.array(inv.get_fdata())

            # Transforming to Axial Manually.

            img_data = np.rollaxis(img_data, self.to_axis, 0)
            label_data = np.rollaxis(label_data, self.to_axis, 0)
            water_data = np.rollaxis(water_data, self.to_axis, 0)
            # fat_data = np.rollaxis(fat_data, self.to_axis, 0)
            # in_data = np.rollaxis(in_data, self.to_axis, 0)

            img_data, _, water_data, label_data, _ = self.remove_black_3channels(img_data, None, water_data, label_data, None)
            # print(img_data.shape)
            # img_data = np.pad(img_data, ((0,0),(1,1),(0,0)), 'constant', constant_values=0)
            # water_data = np.pad(water_data, ((0,0),(1,1),(0,0)), 'constant', constant_values=0)
            # label_data = np.pad(label_data, ((0,0),(1,1),(0,0)), 'constant', constant_values=0)

            # cw, _ = estimate_weights_mfb(label_data)
            # w = estimate_weights_per_slice(label_data)

            img_array.extend(img_data)
            label_array.extend(label_data)
            # print(img_data.shape, water_data.shape)
            water_array.extend(water_data)
            # fat_array.extend(fat_data)
            # in_array.extend(in_data)
            # cw_array.extend(cw)
            # w_array.extend(w)
            img.uncache()
            label.uncache()
            water.uncache()
            # del cw, w

        X = np.stack(img_array, axis=0) if len(img_array) > 1 else img_array[0]
        y = np.stack(label_array, axis=0) if len(label_array) > 1 else label_array[0]
        # indices, az, bz = np.where(y==8)
        # self.gb_min = min(bz)
        # self.gb_max = max(bz)
        water_ = np.stack(water_array, axis=0) if len(water_array) > 1 else water_array[0]
        # fat_ = np.stack(fat_array, axis=0) if len(fat_array) > 1 else fat_array[0]
        # in_ = np.stack(in_array, axis=0) if len(in_array) > 1 else in_array[0]
        # class_weights = np.stack(cw_array, axis=0) if len(cw_array) > 1 else cw_array[0]
        # weights = np.array(w_array)pp5_axi_KORA_mfb_all
        self.y = y   
        self.X = X
        self.water = water_
        # self.fat = fat_
        # self.inv = in_
        self.cw, _ = estimate_weights_mfb(self.y)
        # self.w = weights

        print(self.X.shape, self.y.shape, self.cw.shape, None)#, self.water.shape)
        
    def __getitem__(self, index):
        # if self.is_train and np.random.rand() < 0.75:
        #     index = np.random.randint(self.gb_min, high=self.gb_max)
        img = self.X[index]
        label = self.y[index]

        if self.water_vols is not None:
            img = self.addWater(index, img)

        if self.fat_vols is not None:
            img = self.addFat(index, img)

        if self.in_vols is not None:
            img = self.addIn(index, img)

        if self.thickSlice is not None:
            img = self.thickenTheSlice(index, img)
            
        if self.transforms is not None:
            img, label = self.transforms((img, label))

        img = img if len(img.shape) == 3 else img[np.newaxis, :, :]
        img = torch.from_numpy(img)
        label = torch.from_numpy(label)
        class_weights = torch.from_numpy(self.cw[index])
        # weights = torch.from_numpy(self.w[index])
        return img.type(torch.FloatTensor), label.type(torch.LongTensor), class_weights.type(torch.FloatTensor)

    def remove_black_3channels(self, data,fat,water, labels, inv):
        clean_data,clean_fat,clean_water, clean_labels, clean_in = [], [],[],[], []
        for i, frame in enumerate(labels):
            unique, counts = np.unique(frame, return_counts=True)
            if counts[0] / sum(counts) < .99:
                clean_labels.append(frame)
                clean_data.append(data[i])
                clean_water.append(water[i])
                # clean_fat.append(fat[i])
                # clean_in.append(inv[i])
        return np.array(clean_data), np.array(clean_fat), np.array(clean_water), np.array(clean_labels), np.array(clean_in)

    def thickenSlices(self, indices):
        thickenImages = []
        for i in indices:
            if self.thickSlice:
                thickenImages.append(self.thickenTheSlice(i))
            elif self.water_vols is not None and self.fat_vols is not None and self.in_vols is not None:
                thickenImages.append(self.addIn(i, self.addFat(i, self.addWater(i))))
            elif self.water_vols is not None and self.in_vols is not None:
                thickenImages.append(self.addIn(i, self.addWater(i)))
            elif self.water_vols is not None and self.fat_vols is not None:
                thickenImages.append(self.addFat(i, self.addWater(i)))
            elif self.water_vols is not None:
                thickenImages.append(self.addWater(i))
            elif self.fat_vols is not None:
                thickenImages.append(self.addFat(i))
            elif self.in_vols is not None:
                thickenImages.append(self.addIn(i))
            else:
                print('No thickening')


        return np.array(thickenImages) # np.stack(thickenImages, axis=0)

    def thickenTheSlice(self, index, img=None):
        img = img if img is not None else self.X[index] 
        if index < 2:
            n1, n2 = index, index
        else:
            n1, n2 = index-1, index-2
        
        if index >= self.X.shape[0]-3:
            p1, p2 = index, index
        else:
            p1, p2 = index+1, index+2

        img_n1 = self.X[n1]
        img_n2 = self.X[n2]
        img_p1 = self.X[p1]
        img_p2 = self.X[p2]
        # print(n2, n1, index, p1, p2)
        img_ts = [img_n2, img_n1, img, img_p1, img_p2]
        thickenImg = np.stack(img_ts, axis=0)
        return thickenImg

    def addWater(self, index, img=None):
        img = img if img is not None else self.X[index] 
        wtr = self.water[index]
        img = np.stack([wtr, img], axis=0)
        return img

    def addFat(self, index, img=None):
        img = img if img is not None else self.X[index] 
        ft = self.fat[index]
        # ft = ft[np.newaxis, :, :] if len(img.shape) == 3 else ft
        img = np.stack([img[0],img[1], ft], axis=0)
        return img

    def addIn(self, index, img=None):
        img = img if img is not None else self.X[index] 
        inv = self.inv[index]
        # ft = ft[np.newaxis, :, :] if len(img.shape) == 3 else ft
        # img = np.stack([img[0],img[1], img[2], inv], axis=0)
        img = np.stack([img[0],img[1], inv], axis=0)
        return img

    def getItem(self, index):
        if (self.thickSlice) or (self.water_vols is not None) or (self.fat_vols is not None) or (self.in_vols is not None):
            imgs = self.thickenSlices(index)
        else:
            imgs = self.X[index]

        labels = self.y[index]
        imgs = imgs if len(imgs.shape) == 4 else imgs[:, np.newaxis, :, :]
        return imgs, labels

    def __len__(self):
        return len(self.y)
