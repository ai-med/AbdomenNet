import os

import h5py
import nibabel as nb
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms

from global_vars import *

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


 def apply_split(data_split, data_dir, label_dir):
    file_paths = du.load_file_paths(data_dir, label_dir)
    print("Total no of volumes to process : %d" % len(file_paths))
    train_ratio, test_ratio = data_split.split(",")
    train_len = int((int(train_ratio) / 100) * len(file_paths))
    train_idx = np.random.choice(len(file_paths), train_len, replace=False)
    test_idx = np.array([i for i in range(len(file_paths)) if i not in train_idx])
    train_file_paths = [file_paths[i] for i in train_idx]
    test_file_paths = [file_paths[i] for i in test_idx]
    return train_file_paths, test_file_paths


def _write_h5(data, label, class_weights, weights, f, mode):
    no_slices, H, W = data[0].shape
    no_classes = weights[0].shape[0]
    for i in range(len(weights)):
        no_slices = data[i].shape[0]
        weights[i] = np.broadcast_to(weights[i], (no_slices, no_classes))
    with h5py.File(f[mode]['data'], "w") as data_handle:
        data_handle.create_dataset("data", data=np.concatenate(data).reshape((-1, H, W)))
    with h5py.File(f[mode]['label'], "w") as label_handle:
        label_handle.create_dataset("label", data=np.concatenate(label).reshape((-1, H, W)))
    with h5py.File(f[mode]['weights'], "w") as weights_handle:
        weights_handle.create_dataset("weights", data=np.concatenate(weights))
    with h5py.File(f[mode]['class_weights'], "w") as class_weights_handle:
        class_weights_handle.create_dataset("class_weights", data=np.concatenate(
            class_weights).reshape((-1, H, W)))


def convert_h5(data_dir, label_dir, data_split, train_volumes, test_volumes, val_volumes, f, data_id, remap_config='none',
               orientation=preprocessor.ORIENTATION['coronal']):
    # Data splitting
    if data_split:
        train_file_paths, test_file_paths = apply_split(data_split, data_dir, label_dir)
    elif train_volumes and test_volumes and val_volumes:
        train_file_paths = du.load_file_paths(data_dir, label_dir, data_id, train_volumes)
        test_file_paths = du.load_file_paths(data_dir, label_dir, data_id, test_volumes)
        val_file_paths = du.load_file_paths(data_dir, label_dir, data_id, val_volumes)

    else:
        raise ValueError('You must either provide the split ratio or a train, train dataset list')

    print("Train dataset size: %d, Test dataset size: %d" % (len(train_file_paths), len(test_file_paths)))
    # loading,pre-processing and writing train data
    print("===Train data===")
    #print(train_file_paths)
    data_train, label_train, class_weights_train, weights_train, _ = du.load_dataset(train_file_paths,
                                                                                     orientation,
                                                                                     remap_config=remap_config,
                                                                                     return_weights=True,
                                                                                     reduce_slices=False,
                                                                                     remove_black=False)

    _write_h5(data_train, label_train, class_weights_train, weights_train, f, mode='train')

    # loading,pre-processing and writing test data
    print("===val data===")
    data_test, label_test, class_weights_test, weights_test, _ = du.load_dataset(val_file_paths,
                                                                                 orientation,
                                                                                 remap_config=remap_config,
                                                                                 return_weights=True,
                                                                                 reduce_slices=False,
                                                                                 remove_black=False)

    _write_h5(data_test, label_test, class_weights_test, weights_test, f, mode='val')

    print("===Test data===")
    data_test, label_test, class_weights_test, weights_test, _ = du.load_dataset(test_file_paths,
                                                                                 orientation,
                                                                                 remap_config=remap_config,
                                                                                 return_weights=True,
                                                                                 reduce_slices=False,
                                                                                 remove_black=False)

    _write_h5(data_test, label_test, class_weights_test, weights_test, f, mode='test')