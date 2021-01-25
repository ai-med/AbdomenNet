import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
from utils.evaluator import evaluate, evaluate2view, evaluate_dice_score, compute_vol_bulk, evaluate3view
# from quicknat import QuickNat
from quick_oct import QuickOct as mclass
# from fastSurferCNN import FastSurferCNN
import inspect
from settings import Settings
from solver import Solver
from utils.data_utils import get_imdb_dataset, get_imdb_dataset_3channel, get_imdb_dataset_2channel
from utils.log_utils import LogWriter
from utils.transform import transforms
import logging
import shutil
import glob
import nibabel as nb
import numpy as np
# from create_datasets.commons import MRIDataset
import torch.utils.data as data
# import torchvision.transforms as transforms
from PIL import Image
# import wandb

# cudnn.deterministic = True
# cudnn.benchmark = False

torch.manual_seed(0)
# torch.set_deterministic(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

torch.set_default_tensor_type('torch.FloatTensor')

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

transform = transforms(rotate_prob=0.5, denoise_prob=0.5) # deform_prob=0.5, 

class MRIDataset(data.Dataset):

    def __init__(self, X_files, y_files, transforms=None, thickSlice=None, water_vols=None, fat_vols=None, in_vols=None, orientation='AXI'):
        self.X_files = X_files
        self.y_files = y_files
        self.transforms = transforms
        self.thickSlice = thickSlice
        self.water_vols = water_vols
        self.fat_vols = fat_vols
        self.in_vols = in_vols

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
        fat_array = list()
        in_array = list()
        cw_array = list()
        w_array = list()

        for vol_f, label_f, water_f, fat_f, in_f in zip(self.X_files, self.y_files, self.water_vols, self.fat_vols, self.in_vols):
        # for vol_f, label_f, water_f in zip(self.X_files, self.y_files, self.water_vols):
        # for vol_f, label_f in zip(self.X_files, self.y_files):
            img, label = nb.load(vol_f), nb.load(label_f)
            water = nb.load(water_f)
            fat = nb.load(fat_f)
            inv = nb.load(in_f)

            img_data = np.array(img.get_fdata())
            label_data = np.array(label.get_fdata())
            water_data = np.array(water.get_fdata())
            fat_data = np.array(fat.get_fdata())
            in_data = np.array(inv.get_fdata())

            # Transforming to Axial Manually.

            img_data = np.rollaxis(img_data, self.to_axis, 0)
            label_data = np.rollaxis(label_data, self.to_axis, 0)
            water_data = np.rollaxis(water_data, self.to_axis, 0)
            fat_data = np.rollaxis(fat_data, self.to_axis, 0)
            in_data = np.rollaxis(in_data, self.to_axis, 0)

            img_data, fat_data, water_data, label_data, in_data = self.remove_black_3channels(img_data, fat_data, water_data, label_data, in_data)
            # print(img_data.shape)
            # img_data = np.pad(img_data, ((0,0),(1,1),(0,0)), 'constant', constant_values=0)
            # water_data = np.pad(water_data, ((0,0),(1,1),(0,0)), 'constant', constant_values=0)
            # label_data = np.pad(label_data, ((0,0),(1,1),(0,0)), 'constant', constant_values=0)

            cw, _ = estimate_weights_mfb(label_data)
            w = estimate_weights_per_slice(label_data)

            img_array.extend(img_data)
            label_array.extend(label_data)
            # print(img_data.shape, water_data.shape)
            water_array.extend(water_data)
            fat_array.extend(fat_data)
            in_array.extend(in_data)
            cw_array.extend(cw)
            w_array.extend(w)
            img.uncache()
            label.uncache()
            del cw, w

        X = np.stack(img_array, axis=0) if len(img_array) > 1 else img_array[0]
        y = np.stack(label_array, axis=0) if len(label_array) > 1 else label_array[0]
        water_ = np.stack(water_array, axis=0) if len(water_array) > 1 else water_array[0]
        fat_ = np.stack(fat_array, axis=0) if len(fat_array) > 1 else fat_array[0]
        in_ = np.stack(in_array, axis=0) if len(in_array) > 1 else in_array[0]
        class_weights = np.stack(cw_array, axis=0) if len(cw_array) > 1 else cw_array[0]
        weights = np.array(w_array)
        self.y = y   
        self.X = X 
        self.water = water_
        self.fat = fat_
        self.inv = in_
        self.cw = class_weights
        self.w = weights

        print(self.X.shape, self.y.shape, self.cw.shape, self.w.shape)#, self.water.shape)

    def __getitem__(self, index):
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
        weights = torch.from_numpy(self.w[index])
        return img.type(torch.FloatTensor), label.type(torch.LongTensor), class_weights.type(torch.FloatTensor), weights.type(torch.FloatTensor)

    def remove_black_3channels(self, data,fat,water, labels, inv):
        clean_data,clean_fat,clean_water, clean_labels, clean_in = [], [],[],[], []
        for i, frame in enumerate(labels):
            unique, counts = np.unique(frame, return_counts=True)
            if counts[0] / sum(counts) < .99:
                clean_labels.append(frame)
                clean_data.append(data[i])
                clean_water.append(water[i])
                clean_fat.append(fat[i])
                clean_in.append(inv[i])
        return np.array(clean_data), np.array(clean_fat), np.array(clean_water), np.array(clean_labels), np.array(clean_in)

    def thickenSlices(self, indices):
        thickenImages = []
        for i in indices:
            if self.thickSlice:
                thickenImages.append(self.thickenTheSlice(i))
            elif self.water_vols is not None and self.fat_vols is not None and self.inv is not None:
                thickenImages.append(self.addIn(i, self.addFat(i, self.addWater(i))))
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
        img = np.stack([img[0],img[1], img[2], inv], axis=0)
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


def load_data(data_params):
    print("Loading dataset")
    if data_params['use_3channel'] == True:
        train_data, test_data = get_imdb_dataset_3channel(data_params)
    elif data_params['use_2channel'] == True:
        train_data, test_data = get_imdb_dataset_2channel(data_params)
    else:
        train_data, test_data = get_imdb_dataset(data_params)
    print("Train size: %i" % len(train_data))
    print("Test size: %i" % len(test_data))
    return train_data, test_data


def train(train_params, common_params, data_params, net_params):
    # train_data, test_data = load_data(data_params)
    #
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_params['train_batch_size'], shuffle=True,
    #                                            num_workers=4, pin_memory=True)
    # val_loader = torch.utils.data.DataLoader(test_data, batch_size=train_params['val_batch_size'], shuffle=False,
    #                                          num_workers=4, pin_memory=True)

    arch_file_path = inspect.getfile(mclass)
    setting_path = common_params['setting_path']
    train_volumes = sorted(glob.glob(f"{data_params['data_dir']}/train/volume/**.nii.gz"))
    train_w_volumes = sorted(glob.glob(f"{data_params['data_dir']}/train/volume_w/**.nii.gz"))
    train_f_volumes = sorted(glob.glob(f"{data_params['data_dir']}/train/volume_f/**.nii.gz"))
    train_in_volumes = sorted(glob.glob(f"{data_params['data_dir']}/train/volume_in/**.nii.gz"))
    train_labels = sorted(glob.glob(f"{data_params['data_dir']}/train/label/**.nii.gz"))

    test_volumes = sorted(glob.glob(f"{data_params['data_dir']}/test/volume/**.nii.gz"))
    test_w_volumes = sorted(glob.glob(f"{data_params['data_dir']}/test/volume_w/**.nii.gz"))
    test_f_volumes = sorted(glob.glob(f"{data_params['data_dir']}/test/volume_f/**.nii.gz"))
    test_in_volumes = sorted(glob.glob(f"{data_params['data_dir']}/test/volume_in/**.nii.gz"))
    test_labels = sorted(glob.glob(f"{data_params['data_dir']}/test/label/**.nii.gz"))

    ds_train = MRIDataset(train_volumes, train_labels, transforms=transform, thickSlice=None, water_vols=train_w_volumes, fat_vols=train_f_volumes, in_vols = train_in_volumes, orientation='AXI')
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=train_params['train_batch_size'], shuffle=True,
                                               num_workers=4, pin_memory=True)

    ds_test = MRIDataset(test_volumes, test_labels, transforms=None, thickSlice=None, water_vols=test_w_volumes, fat_vols=test_f_volumes, in_vols = test_in_volumes, orientation='AXI')
    val_loader = torch.utils.data.DataLoader(ds_test, batch_size=train_params['val_batch_size'], shuffle=False,
                                             num_workers=4, pin_memory=True)

    if train_params['use_pre_trained']:
        model = torch.load(train_params['pre_trained_path'])
    else:
        net_params_ = net_params.copy()
        if net_params['type'] == 'quicknat':
            model = mclass(net_params)
            empty_model = mclass(net_params_)
        # elif net_params['type'] == 'fastsurfer':
        #     model = FastSurferCNN(net_params)

       # {"lr": train_params['learning_rate'],
    #   "momentum": train_params['momentum'],
    #   "weight_decay": train_params['optim_weight_decay']},
# {"lr": train_params['learning_rate'],
#                                    "betas": train_params['optim_betas'],
#                                    "eps": train_params['optim_eps'],
#                                    "weight_decay": train_params['optim_weight_decay']},

    # wandb.watch(model)

    solver = Solver(model,
                    device=common_params['device'],
                    num_class=net_params['num_class'],
                    optim_args={"lr": train_params['learning_rate']},
                    model_name=common_params['model_name'],
                    exp_name=train_params['exp_name'],
                    labels=data_params['labels'],
                    log_nth=train_params['log_nth'],
                    num_epochs=train_params['num_epochs'],
                    lr_scheduler_step_size=train_params['lr_scheduler_step_size'],
                    lr_scheduler_gamma=train_params['lr_scheduler_gamma'],
                    use_last_checkpoint=train_params['use_last_checkpoint'],
                    log_dir=common_params['log_dir'],
                    exp_dir=common_params['exp_dir'],
                    arch_file_path=[arch_file_path,setting_path])

    solver.train(train_loader, val_loader)
    final_model_path = os.path.join(common_params['save_model_dir'], train_params['final_model_file'])
    # model.save(final_model_path)
    solver.model = empty_model
    solver.save_best_model(final_model_path)
    print("final model saved @ " + str(final_model_path))


def evaluate(eval_params, net_params, data_params, common_params, train_params):
    eval_model_path = eval_params['eval_model_path']
    num_classes = net_params['num_class']
    labels = data_params['labels']
    data_dir = eval_params['data_dir']
    label_dir = eval_params['label_dir']
    volumes_txt_file = eval_params['volumes_txt_file']
    remap_config = eval_params['remap_config']
    device = common_params['device']
    log_dir = common_params['log_dir']
    exp_dir = common_params['exp_dir']
    exp_name = train_params['exp_name']
    save_predictions_dir = eval_params['save_predictions_dir']
    prediction_path = os.path.join(exp_dir, exp_name, save_predictions_dir)
    orientation = eval_params['orientation']
    data_id = eval_params['data_id']
    multi_channel = data_params['use_3channel']
    use_2channel = data_params['use_2channel']
    thick_channel = data_params['thick_channel']
    logWriter = LogWriter(num_classes, log_dir, exp_name, labels=labels)

    avg_dice_score, class_dist = evaluate_dice_score(eval_model_path,
                                                        num_classes,
                                                        data_dir,
                                                        label_dir,
                                                        volumes_txt_file,
                                                        remap_config,
                                                        orientation,
                                                        prediction_path,
                                                        data_id,
                                                        device,
                                                        logWriter,
                                                        multi_channel=multi_channel,
                                                        use_2channel=use_2channel,
                                                        thick_ch=thick_channel)
    logWriter.close()


def evaluate_bulk(eval_bulk):
    data_dir = eval_bulk['data_dir']
    label_dir = eval_bulk['label_dir']
    label_list = eval_bulk['label_list']
    prediction_path = eval_bulk['save_predictions_dir']
    volumes_txt_file = eval_bulk['volumes_txt_file']
    device = eval_bulk['device']
    label_names = [ 'vol_ID','liver', 'spleen', 'kidney_r', 'kidney_l', 'adrenal_r', 'adrenal_l', 'pancreas', 'gallbladder']
    batch_size = eval_bulk['batch_size']
    need_unc = eval_bulk['estimate_uncertainty']
    mc_samples = eval_bulk['mc_samples']
    dir_struct = eval_bulk['directory_struct']
    multi_channel = data_params['use_3channel']
    use_2channel = data_params['use_2channel']

    if 'exit_on_error' in eval_bulk.keys():
        exit_on_error = eval_bulk['exit_on_error']
    else:
        exit_on_error = False

    if eval_bulk['view_agg'] == 'True':
        coronal_model_path = eval_bulk['coronal_model_path']
        axial_model_path = eval_bulk['axial_model_path']
        evaluate2view(
            coronal_model_path,
            axial_model_path,
            volumes_txt_file,
            data_dir, label_dir, device,
            prediction_path,
            batch_size,
            label_names,
            label_list,
            dir_struct,
            need_unc,
            mc_samples,
            exit_on_error=exit_on_error,
            multi_channel=multi_channel,
            use_2channel=use_2channel
        )
    elif eval_bulk['3view_agg'] == 'True':
        coronal_model_path = eval_bulk['coronal_model_path']
        axial_model_path = eval_bulk['axial_model_path']
        sagittal_model_path = eval_bulk['sagittal_model_path']
        evaluate3view(
            coronal_model_path,
            axial_model_path,
            sagittal_model_path,
            volumes_txt_file,
            data_dir, label_dir, device,
            prediction_path,
            batch_size,
            label_names,
            label_list,
            dir_struct,
            need_unc,
            mc_samples,
            exit_on_error=exit_on_error,
            multi_channel=multi_channel,
            use_2channel=use_2channel
        )
    else:
        coronal_model_path = eval_bulk['coronal_model_path']
        evaluate(
            coronal_model_path,
            volumes_txt_file,
            data_dir,
            label_dir, label_list,
            device,
            prediction_path,
            batch_size,
            "COR",
            label_names,
            dir_struct,
            need_unc,
            mc_samples,
            exit_on_error=exit_on_error,
            multi_channel=multi_channel
        )

def compute_vol(eval_bulk):
    prediction_path = eval_bulk['save_predictions_dir']
    label_names = ['liver', 'spleen', 'kidney_r', 'kidney_l', 'adrenal_r', 'adrenal_l', 'pancreas', 'gallbladder']
    volumes_txt_file = eval_bulk['volumes_txt_file']

    compute_vol_bulk(prediction_path, "Linear", label_names, volumes_txt_file)



def delete_contents(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', required=True, help='run mode, valid values are train and eval')
    parser.add_argument('--setting_path', '-sp', required=False, default=None, help='optional path to settings_eval_nako.ini')
    args = parser.parse_args()
    if args.setting_path is None:
        args.setting_path = '/home/abhijit/Jyotirmay/abdominal_segmentation/quickNAT_pytorch/settings_merged_jj.ini'

    settings = Settings(args.setting_path)

    
    exp_name = settings['TRAINING']['exp_name']
    print(exp_name)
    # wandb.login(key='85588d16512e76335148306ccccf96e543c5f627')
    # wandb.init(name='r0-run', project=settings['TRAINING']['exp_name'], config=settings)

    common_params, data_params, net_params, train_params, eval_params = settings['COMMON'], settings['DATA'], \
                                                                        settings[
                                                                            'NETWORK'], settings['TRAINING'], \
                                                                        settings['EVAL']
    common_params['setting_path'] = args.setting_path
    if args.mode == 'train':
        train(train_params, common_params, data_params, net_params)
    elif args.mode == 'eval':
        evaluate(eval_params, net_params, data_params, common_params, train_params)
    elif args.mode == 'eval_bulk':
        logging.basicConfig(filename='error.log')
        settings_eval = Settings(args.setting_path)
        evaluate_bulk(settings_eval['EVAL_BULK'])
    elif args.mode == 'clear':
        shutil.rmtree(os.path.join(common_params['exp_dir'], train_params['exp_name']))
        print("Cleared current experiment directory successfully!!")
        shutil.rmtree(os.path.join(common_params['log_dir'], train_params['exp_name']))
        print("Cleared current log directory successfully!!")

    elif args.mode == 'clear-all':
        delete_contents(common_params['exp_dir'])
        print("Cleared experiments directory successfully!!")
        delete_contents(common_params['log_dir'])
        print("Cleared logs directory successfully!!")

    elif args.mode == 'compute_vol':
        if args.setting_path is not None:
            settings_eval = Settings(args.setting_path)
        else:
            settings_eval = Settings('settings_eval_ukb.ini')
        compute_vol(settings_eval['EVAL_BULK'])
    else:
        raise ValueError('Invalid value for mode. only support values are train, eval and clear')
