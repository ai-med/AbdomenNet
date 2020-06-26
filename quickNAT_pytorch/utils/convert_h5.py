"""
Convert to h5 utility.
Sample command to create new dataset
- python3 utils/convert_h5.py -dd /home/masterthesis/shayan/nas_drive/Data_Neuro/OASISchallenge/FS -ld /home/masterthesis/shayan/nas_drive/Data_Neuro/OASISchallenge -trv create_datasets/train_volumes.txt -tev create_datasets/test_volumes.txt -id MALC -rc Neo -o COR -df create_datasets/MALC/coronal
- python utils/convert_h5.py -dd /home/masterthesis/shayan/nas_drive/Data_Neuro/IXI/IXI_FS -ld /home/masterthesis/shayan/nas_drive/Data_Neuro/IXI/IXI_FS -ds 98,2 -rc FS -o COR -df create_datasets/IXI/coronal

for single create_datasets:

python utils/convert_h5.py -dd /home/anne/phd/projects/whole_body/whole_body_segmentation/quickNAT_pytorch/create_datasets/KORA/data -ld /home/anne/phd/projects/whole_body/whole_body_segmentation/quickNAT_pytorch/create_datasets/KORA/data  -rc none -o COR -df /home/anne/phd/projects/whole_body/whole_body_segmentation/quickNAT_pytorch/create_datasets/KORA/coronal_2 -id KORA -trv /home/anne/phd/projects/whole_body/whole_body_segmentation/quickNAT_pytorch/create_datasets/KORA/KORA.train -tev /home/anne/phd/projects/whole_body/whole_body_segmentation/quickNAT_pytorch/create_datasets/KORA/KORA.test -valv /home/anne/phd/projects/whole_body/whole_body_segmentation/quickNAT_pytorch/create_datasets/new/KORA/KORA.val

 -dd /mnt/nas/Users/Anne/wholebodysegmentation/datasets/multi_channel/ -ld /mnt/nas/Users/Anne/wholebodysegmentation/datasets/multi_channel/  -rc none -o SAG -df /mnt/nas/Users/Anne/wholebodysegmentation/datasets/multi_channel/KORA/sagittal -id KORA -trv /mnt/nas/Users/Anne/wholebodysegmentation/datasets/multi_channel/KORA/KORA.train -tev /mnt/nas/Users/Anne/wholebodysegmentation/datasets/multi_channel/KORA/KORA.test -valv /mnt/nas/Users/Anne/wholebodysegmentation/datasets/multi_channel/KORA/KORA.val -nc 3

python utils/convert_h5.py -dd /mnt/nas/Users/Anne/wholebodysegmentation/datasets -ld /mnt/nas/Users/Anne/wholebodysegmentation/datasets  -rc none -o SAG -df /mnt/nas/Users/Anne/wholebodysegmentation/datasets/KORANAKOUKB/sagittal -id All


"""

import argparse
import os

import h5py
import numpy as np

import common_utils
import data_utils as du
import preprocessor as preprocessor


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

def _write_h5_3channel(data, fat, water, label, class_weights, weights, f, mode):
    no_slices, H, W = data[0].shape
    no_classes = weights[0].shape[0]
    for i in range(len(weights)):
        no_slices = data[i].shape[0]
        weights[i] = np.broadcast_to(weights[i], (no_slices, no_classes))
    with h5py.File(f[mode]['fat'], "w") as data_handle:
        data_handle.create_dataset("data", data=np.concatenate(fat).reshape((-1, H, W)))

    with h5py.File(f[mode]['water'], "w") as data_handle:
        data_handle.create_dataset("data", data=np.concatenate(water).reshape((-1, H, W)))

    with h5py.File(f[mode]['data'], "w") as data_handle:
        data_handle.create_dataset("data", data=np.concatenate(data).reshape((-1, H, W)))

    with h5py.File(f[mode]['label'], "w") as label_handle:
        label_handle.create_dataset("label", data=np.concatenate(label).reshape((-1, H, W)))
    with h5py.File(f[mode]['weights'], "w") as weights_handle:
        weights_handle.create_dataset("weights", data=np.concatenate(weights))
    with h5py.File(f[mode]['class_weights'], "w") as class_weights_handle:
        class_weights_handle.create_dataset("class_weights", data=np.concatenate(
            class_weights).reshape((-1, H, W)))

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

def convert_h5_3channel(data_dir, label_dir, data_split, train_volumes, test_volumes, val_volumes, f, data_id, remap_config='none',
               orientation=preprocessor.ORIENTATION['coronal']):
    # Data splitting
    if data_split:
        train_file_paths, test_file_paths = apply_split(data_split, data_dir, label_dir)
    elif train_volumes and test_volumes and val_volumes:
        train_file_paths = du.load_file_paths_3channel(data_dir, label_dir, data_id, train_volumes)
        test_file_paths = du.load_file_paths_3channel(data_dir, label_dir, data_id, test_volumes)
        val_file_paths = du.load_file_paths_3channel(data_dir, label_dir, data_id, val_volumes)

    else:
        raise ValueError('You must either provide the split ratio or a train, train dataset list')

    print("Train dataset size: %d, Test dataset size: %d" % (len(train_file_paths), len(test_file_paths)))
    # loading,pre-processing and writing train data
    print("===Train data===")
    #print(train_file_paths)
    data_train, fat_train, water_train, label_train, class_weights_train, weights_train, _ = du.load_dataset_3channel(train_file_paths,
                                                                                     orientation,
                                                                                     remap_config=remap_config,
                                                                                     return_weights=True,
                                                                                     reduce_slices=False,
                                                                                     remove_black=False)

    _write_h5_3channel(data_train,fat_train, water_train, label_train, class_weights_train, weights_train, f, mode='train')

    # loading,pre-processing and writing test data
    print("===val data===")
    data_test, fat_test, water_test, label_test, class_weights_test, weights_test, _ = du.load_dataset_3channel(val_file_paths,
                                                                                 orientation,
                                                                                 remap_config=remap_config,
                                                                                 return_weights=True,
                                                                                 reduce_slices=False,
                                                                                 remove_black=False)

    _write_h5_3channel(data_test, fat_test, water_test, label_test, class_weights_test, weights_test, f, mode='val')

    print("===Test data===")
    data_test,fat_test, water_test, label_test, class_weights_test, weights_test, _ = du.load_dataset_3channel(test_file_paths,
                                                                                 orientation,
                                                                                 remap_config=remap_config,
                                                                                 return_weights=True,
                                                                                 reduce_slices=False,
                                                                                 remove_black=False)

    _write_h5_3channel(data_test, fat_test, water_test, label_test, class_weights_test, weights_test, f, mode='test')

def convert_h5_multi(data_dir, label_dir, data_split, train_volumes, test_volumes, val_volumes, f, data_id, remap_config='none',
               orientation=preprocessor.ORIENTATION['coronal']):

    if data_id =='All':
        all_data_train = []
        all_label_train = []
        all_class_weights_train = []
        all_weights_train = []

        all_data_test = []
        all_label_test = []
        all_class_weights_test = []
        all_weights_test = []

        all_data_val = []
        all_label_val = []
        all_class_weights_val = []
        all_weights_val = []

        for did in ['KORA','NAKO','UKB']:
            train_file_paths = du.load_file_paths(os.path.join(data_dir, did, 'data'), os.path.join(label_dir, did, 'data'), did, os.path.join(data_dir, did, did+'.train'))
            test_file_paths = du.load_file_paths(os.path.join(data_dir, did, 'data'), os.path.join(label_dir, did, 'data'), did, os.path.join(data_dir, did, did+'.test'))
            val_file_paths = du.load_file_paths(os.path.join(data_dir, did, 'data'), os.path.join(label_dir, did, 'data'), did, os.path.join(data_dir, did, did+'.val'))
            print("Train dataset size: %d, Test dataset size: %d" % (len(train_file_paths), len(test_file_paths)))
            # loading,pre-processing and writing train data
            print("===Train data===")
            data_train, label_train, class_weights_train, weights_train, _ = du.load_dataset(train_file_paths,
                                                                                     orientation,
                                                                                     remap_config=remap_config,
                                                                                     return_weights=True,
                                                                                     reduce_slices=False,
                                                                                     remove_black=False)
            all_data_train += data_train
            all_label_train += label_train
            all_class_weights_train += class_weights_train
            all_weights_train += weights_train

            print("===val data===")
            data_val, label_val, class_weights_val, weights_val, _ = du.load_dataset(val_file_paths,
                                                                                         orientation,
                                                                                         remap_config=remap_config,
                                                                                         return_weights=True,
                                                                                         reduce_slices=False,
                                                                                         remove_black=False)

            all_data_val += data_val
            all_label_val += label_val
            all_class_weights_val += class_weights_val
            all_weights_val += weights_val

            print("===Test data===")
            data_test, label_test, class_weights_test, weights_test, _ = du.load_dataset(test_file_paths,
                                                                                         orientation,
                                                                                         remap_config=remap_config,
                                                                                         return_weights=True,
                                                                                         reduce_slices=False,
                                                                                         remove_black=False)
            all_data_test += data_test
            all_label_test += label_test
            all_class_weights_test += class_weights_test
            all_weights_test += weights_test

        print("===Save train data===")
        _write_h5(all_data_train, all_label_train, all_class_weights_train, all_weights_train, f, mode='train')

        # loading,pre-processing and writing test data
        print("===Save val data===")
        _write_h5(all_data_val, all_label_val, all_class_weights_val, all_weights_val, f, mode='val')
        print("===Save test data===")
        _write_h5(all_data_test, all_label_test, all_class_weights_test, all_weights_test, f, mode='test')
    else:
        raise ValueError('You must either provide the split ratio or a train, train dataset list')





if __name__ == "__main__":
    print("* Start *")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-dd', required=True,
                        help='Base directory of the data folder. This folder should contain one folder per volume.')
    parser.add_argument('--label_dir', '-ld', required=True,
                        help='Base directory of all the label files. This folder should have one file per volumn with same name as the corresponding volumn folder name inside data_dir')
    parser.add_argument('--data_split', '-ds', required=False,
                        help='Ratio to split data randomly into train and test. input e.g. 80,20')
    parser.add_argument('--train_volumes', '-trv', required=False,
                        help='Path to a text file containing the list of volumes to be used for training')
    parser.add_argument('--test_volumes', '-tev', required=False,
                        help='Path to a text file containing the list of volumes to be used for testing')
    parser.add_argument('--val_volumes', '-valv', required=False,
                        help='Path to a text file containing the list of volumes to be used for validation')
    parser.add_argument('--data_id', '-id', required=True, help='Valid options are "MALC", "ADNI", "CANDI" and "IBSR"')
    parser.add_argument('--remap_config', '-rc', required=True, help='Valid options are "FS", "Neo" and "FS_parcel"')
    parser.add_argument('--orientation', '-o', required=True, help='Valid options are COR, AXI, SAG')
    parser.add_argument('--destination_folder', '-df', required=True, help='Path where to generate the h5 files')
    parser.add_argument('--num_channels', '-nc', required=False, help='Number of input channels, default = 1, set to 3 for opp, fat and water')
    args = parser.parse_args()

    common_utils.create_if_not(args.destination_folder)

    f = {
        'train': {
            "data": os.path.join(args.destination_folder, "Data_train.h5"),
            "fat": os.path.join(args.destination_folder, "Data_fat_train.h5"),
            "water": os.path.join(args.destination_folder, "Data_water_train.h5"),
            "label": os.path.join(args.destination_folder, "Label_train.h5"),
            "weights": os.path.join(args.destination_folder, "Weight_train.h5"),
            "class_weights": os.path.join(args.destination_folder, "Class_Weight_train.h5"),
        },
        'test': {
            "data": os.path.join(args.destination_folder, "Data_test.h5"),
            "fat": os.path.join(args.destination_folder, "Data_fat_test.h5"),
            "water": os.path.join(args.destination_folder, "Data_water_test.h5"),
            "label": os.path.join(args.destination_folder, "Label_test.h5"),
            "weights": os.path.join(args.destination_folder, "Weight_test.h5"),
            "class_weights": os.path.join(args.destination_folder, "Class_Weight_test.h5")
        },
        'val': {
            "data": os.path.join(args.destination_folder, "Data_val.h5"),
            "fat": os.path.join(args.destination_folder, "Data_fat_val.h5"),
            "water": os.path.join(args.destination_folder, "Data_water_val.h5"),
            "label": os.path.join(args.destination_folder, "Label_val.h5"),
            "weights": os.path.join(args.destination_folder, "Weight_val.h5"),
            "class_weights": os.path.join(args.destination_folder, "Class_Weight_val.h5")
        }
    }
    print(args.data_id)
    print('num channels ', args.num_channels)
    if args.num_channels == '3':
        print('convert single, multi cchannel')
        convert_h5_3channel(args.data_dir, args.label_dir, args.data_split, args.train_volumes, args.test_volumes,
                   args.val_volumes, f,
                   args.data_id,
                   args.remap_config,
                   args.orientation)
    elif args.data_id == 'All':
        print('convert multi')
        convert_h5_multi(args.data_dir, args.label_dir, args.data_split, args.train_volumes, args.test_volumes,
                   args.val_volumes, f,
                   args.data_id,
                   args.remap_config,
                   args.orientation)
    else:
        print('convert single')
        convert_h5(args.data_dir, args.label_dir, args.data_split, args.train_volumes, args.test_volumes, args.val_volumes, f,
                   args.data_id,
                   args.remap_config,
                   args.orientation)
    print("* Finish *")
