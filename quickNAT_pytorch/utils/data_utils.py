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

def estimate_weights_per_slice(labels, no_of_class=9):
    weights_per_slice = []
    for slice_ in labels:
        unique, counts = np.unique(slice_, return_counts=True)
        median_freq = np.median(counts)
        weights = np.zeros(no_of_class)
        for i, label in enumerate(unique):
            weights[int(label)] = median_freq // counts[i]
        weights_per_slice.append(weights)

    return np.array(weights_per_slice)

class MRIDataset(data.Dataset):

    def __init__(self, X_files, y_files, transforms=None, thickSlice=None, water_vols=None, fat_vols=None, orientation='AXI'):
        self.X_files = X_files
        self.y_files = y_files
        self.transforms = transforms
        self.thickSlice = thickSlice
        self.water_vols = water_vols
        self.fat_vols = fat_vols

        if orientation == 'AXI':
            self.to_axis = 2
        elif orientation == 'COR':
            self.to_axis = 1
        else:
            self.to_axis = 0

        img_array = list()
        label_array = list()
        water_array = list()
        # fat_array = list()
        cw_array = list()
        w_array = list()

        # for vol_f, label_f, water_f, fat_f in zip(self.X_files, self.y_files, self.water_vols, self.fat_vols):
        for vol_f, label_f, water_f in zip(self.X_files, self.y_files, self.water_vols):
        # for vol_f, label_f in zip(self.X_files, self.y_files):
            img, label = nb.load(vol_f), nb.load(label_f)
            water = nb.load(water_f)
            # fat = nb.load(fat_f)

            img_data = np.array(img.get_fdata())
            label_data = np.array(label.get_fdata())
            water_data = np.array(water.get_fdata())
            # fat_data = np.array(fat.get_fdata())

            # Transforming to target axes.
            img_data = np.rollaxis(img_data, self.to_axis, 0)
            label_data = np.rollaxis(label_data, self.to_axis, 0)
            water_data = np.rollaxis(water_data, self.to_axis, 0)
            # fat_data = np.rollaxis(fat_data, self.to_axis, 0)

            img_data, _, water_data, label_data = self.remove_black_3channels(img_data, None, water_data, label_data)

            cw, _ = estimate_weights_mfb(label_data)
            w = estimate_weights_per_slice(label_data)

            img_array.extend(img_data)
            label_array.extend(label_data)
            water_array.extend(water_data)
            # fat_array.extend(fat_data)
            cw_array.extend(cw)
            w_array.extend(w)
            img.uncache()
            label.uncache()
            del cw, w

        X = np.stack(img_array, axis=0) if len(img_array) > 1 else img_array[0]
        y = np.stack(label_array, axis=0) if len(label_array) > 1 else label_array[0]
        water_ = np.stack(water_array, axis=0) if len(water_array) > 1 else water_array[0]
        fat_ = None#np.stack(fat_array, axis=0) if len(fat_array) > 1 else fat_array[0]
        class_weights = np.stack(cw_array, axis=0) if len(cw_array) > 1 else cw_array[0]
        weights = np.array(w_array)
        self.y = y   
        self.X = X 
        self.water = water_
        self.fat = fat_
        self.cw = class_weights
        self.w = weights

        print(self.X.shape, self.y.shape, self.cw.shape, self.w.shape)

    def __getitem__(self, index):
        img = self.X[index]
        label = self.y[index]

        if self.water_vols is not None:
            img = self.addWater(index, img)

        if self.fat_vols is not None:
            img = self.addFat(index, img)

        if self.thickSlice is not None:
            img = self.thickenTheSlice(index, img)
            
        if self.transforms is not None:
            img, label = self.transforms((img, label))

        img = img if len(img.shape) == 3 else img[np.newaxis, :, :]
        img = torch.from_numpy(img)
        label = torch.from_numpy(label)
        class_weights = torch.from_numpy(self.cw[index])
        weights = torch.from_numpy(self.w[index])
        return img.type(torch.FloatTensor), label.type(torch.LongTensor), class_weights.type(torch.FloatTensor), weights.type(torch.FloatTensor)

    def remove_black_3channels(self, data,fat,water, labels):
        clean_data,clean_fat,clean_water, clean_labels = [], [],[],[]
        for i, frame in enumerate(labels):
            unique, counts = np.unique(frame, return_counts=True)
            if counts[0] / sum(counts) < .99:
                clean_labels.append(frame)
                clean_data.append(data[i])
                clean_water.append(water[i])
                # clean_fat.append(fat[i])
        return np.array(clean_data), np.array(clean_fat), np.array(clean_water), np.array(clean_labels)

    def thickenSlices(self, indices):
        thickenImages = []
        for i in indices:
            if self.thickSlice:
                thickenImages.append(self.thickenTheSlice(i))
            elif self.water_vols is not None and self.fat_vols is not None:
                thickenImages.append(self.addFat(i, self.addWater(i)))
            elif self.water_vols is not None:
                thickenImages.append(self.addWater(i))
            elif self.fat_vols is not None:
                thickenImages.append(self.addFat(i))
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
        img = np.stack([img[0],img[1], ft], axis=0)
        return img

    def getItem(self, index):
        if (self.thickSlice) or (self.water_vols is not None) or (self.fat_vols is not None):
            imgs = self.thickenSlices(index)
        else:
            imgs = self.X[index]

        labels = self.y[index]
        imgs = imgs if len(imgs.shape) == 4 else imgs[:, np.newaxis, :, :]
        return imgs, labels

    def __len__(self):
        return len(self.y)
