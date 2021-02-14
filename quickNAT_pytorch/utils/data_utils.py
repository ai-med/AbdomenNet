import os
import nibabel as nb
import numpy as np
import torch
import torch.utils.data as data

def load_file_paths_3channel(data_dir, label_dir, volumes_txt_file=None):
    """
    This function returns the file paths combined as a list where each element is a 2 element tuple, 0th being data and 1st being label.
    It should be modified to suit the need of the project
    :param data_dir: Directory which contains the data files
    :param label_dir: Directory which contains the label files
    :param volumes_txt_file: (Optional) Path to the a csv file, when provided only these data points will be read
    :return: list of file paths as string
    """

    if volumes_txt_file:
        with open(volumes_txt_file) as file_handle:
            volumes_to_use = file_handle.read().splitlines()
    else:
        volumes_to_use = [name for name in os.listdir(data_dir)]

    file_paths = [
        [os.path.join(data_dir, f'{vol}.nii.gz'), 
         os.path.join(label_dir, f'{vol}.nii.gz'),
         os.path.join(f'{data_dir}_w', f'{vol}.nii.gz'), 
         os.path.join(f'{data_dir}_f', f'{vol}.nii.gz'), 
         os.path.join(f'{data_dir}_in', f'{vol}.nii.gz')]
        for
        vol in volumes_to_use]

    return file_paths

def load_file_paths(data_dir, label_dir, volumes_txt_file=None):
    """
    This function returns the file paths combined as a list where each element is a 2 element tuple, 0th being data and 1st being label.
    It should be modified to suit the need of the project
    :param data_dir: Directory which contains the data files
    :param label_dir: Directory which contains the label files
    :param volumes_txt_file: (Optional) Path to the a csv file, when provided only these data points will be read
    :return: list of file paths as string
    """

    if volumes_txt_file:
        with open(volumes_txt_file) as file_handle:
            volumes_to_use = file_handle.read().splitlines()
    else:
        volumes_to_use = [name for name in os.listdir(data_dir)]

    file_paths = [
            [os.path.join(data_dir, f'{vol}.nii.gz'), os.path.join(label_dir, f'{vol}.nii.gz')]
            for
            vol in volumes_to_use]

    return file_paths