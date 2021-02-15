import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
from utils.evaluator import evaluate_dice_score, evaluate3view
# from quicknat import QuickNat as mclass
from quick_oct import QuickOct as mclass
import inspect
from settings import Settings
from solver_sgd import Solver
from utils.data_utils import MRIDataset

from utils.log_utils import LogWriter, telegram_notifier
from utils.transform import transforms
import logging
import shutil
import glob
import nibabel as nb
import numpy as np
import torch.utils.data as data
from PIL import Image

# torch.manual_seed(0)
# # torch.set_deterministic(True)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(0)

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

            # Transforming to Axial Manually.

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
        fat_ = None #np.stack(fat_array, axis=0) if len(fat_array) > 1 else fat_array[0]
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
                if water is not None:
                    clean_water.append(water[i])
                if fat is not None:
                    clean_fat.append(fat[i])
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


        return np.array(thickenImages)

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
        # ft = ft[np.newaxis, :, :] if len(img.shape) == 3 else ft
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

def train(train_params, common_params, data_params, net_params):

    arch_file_path = inspect.getfile(mclass)
    setting_path = common_params['setting_path']
    train_volumes = sorted(glob.glob(f"{data_params['data_dir']}/volume/**.nii.gz"))
    train_w_volumes = sorted(glob.glob(f"{data_params['data_dir']}/volume_w/**.nii.gz"))
    # train_f_volumes = sorted(glob.glob(f"{data_params['data_dir']}/volume_f/**.nii.gz"))
    train_labels = sorted(glob.glob(f"{data_params['data_dir']}/label/**.nii.gz"))

    test_volumes = sorted(glob.glob(f"{data_params['val_dir']}/volume/**.nii.gz"))
    test_w_volumes = sorted(glob.glob(f"{data_params['val_dir']}/volume_w/**.nii.gz"))
    # test_f_volumes = sorted(glob.glob(f"{data_params['val_dir']}/volume_f/**.nii.gz"))
    test_labels = sorted(glob.glob(f"{data_params['val_dir']}/label/**.nii.gz"))

    ds_train = MRIDataset(train_volumes, train_labels, transforms=transform, thickSlice=None, water_vols=train_w_volumes,
     fat_vols=None, orientation=train_params['orientation'])
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=train_params['train_batch_size'], shuffle=True,
                                               num_workers=4, pin_memory=True)

    ds_test = MRIDataset(test_volumes, test_labels, transforms=None, thickSlice=None, water_vols=test_w_volumes, 
    fat_vols=None, orientation=train_params['orientation'])
    val_loader = torch.utils.data.DataLoader(ds_test, batch_size=train_params['val_batch_size'], shuffle=False,
                                             num_workers=4, pin_memory=True)
    net_params_ = net_params.copy()

    if train_params['use_pre_trained']:
        model = torch.load(train_params['pre_trained_path'])
    else:
        if net_params['type'] == 'quicknat':
            model = mclass(net_params)

    empty_model = mclass(net_params_)

    solver = Solver(model,
                    device=common_params['device'],
                    num_class=net_params['num_class'],
                    optim=torch.optim.SGD,
                    optim_args={"lr": train_params['learning_rate'], 
                    "momentum": train_params['momentum'],
                    "weight_decay": train_params['optim_weight_decay']},
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

    solver.model = empty_model
    solver.save_best_model(final_model_path)
    print("final model saved @ " + str(final_model_path))
    telegram_notifier(f"{final_model_path} Done!!")


def evaluate(eval_params, net_params, data_params, common_params, train_params):
    eval_model_path = eval_params['eval_model_path']
    num_classes = net_params['num_class']
    labels = data_params['labels']
    data_dir = eval_params['data_dir']
    label_dir = eval_params['label_dir']
    volumes_txt_file = eval_params['volumes_txt_file']
    device = common_params['device']
    log_dir = common_params['log_dir']
    exp_dir = common_params['exp_dir']
    exp_name = train_params['exp_name']
    save_predictions_dir = eval_params['save_predictions_dir']
    prediction_path = os.path.join(exp_dir, exp_name, save_predictions_dir)
    orientation = eval_params['orientation']
    multi_channel = data_params['use_3channel']
    use_2channel = data_params['use_2channel']
    thick_channel = data_params['thick_channel']
    logWriter = LogWriter(num_classes, log_dir, exp_name, labels=labels)

    avg_dice_score, class_dist = evaluate_dice_score(eval_model_path,
                                                        num_classes,
                                                        data_dir,
                                                        label_dir,
                                                        volumes_txt_file,
                                                        orientation,
                                                        prediction_path,
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
    label_names = eval_bulk['label_names']
    batch_size = eval_bulk['batch_size']
    multi_channel = data_params['use_3channel']
    use_2channel = data_params['use_2channel']

    if 'exit_on_error' in eval_bulk.keys():
        exit_on_error = eval_bulk['exit_on_error']
    else:
        exit_on_error = False

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
        exit_on_error=exit_on_error,
        multi_channel=multi_channel,
        use_2channel=use_2channel
    )

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
        args.setting_path = '/home/jyotirmay/remote_projects/abdominal_segmentation_2/quickNAT_pytorch/settings.ini'

    settings = Settings(args.setting_path)
    
    exp_name = settings['TRAINING']['exp_name']
    print(exp_name)

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

    else:
        raise ValueError('Invalid value for mode. only support values are train, eval, eval_bulk and clear')