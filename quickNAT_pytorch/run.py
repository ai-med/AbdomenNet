import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

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
from utils.data_utils import MRIDataset
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
    # train_f_volumes = sorted(glob.glob(f"{data_params['data_dir']}/train/volume_f/**.nii.gz"))
    # train_in_volumes = sorted(glob.glob(f"{data_params['data_dir']}/train/volume_in/**.nii.gz"))
    train_labels = sorted(glob.glob(f"{data_params['data_dir']}/train/label/**.nii.gz"))
    orientation = 'SAG'
    # test_volumes = sorted(glob.glob(f"{data_params['data_dir']}/test/volume/**.nii.gz"))
    # test_w_volumes = sorted(glob.glob(f"{data_params['data_dir']}/test/volume_w/**.nii.gz"))
    # # test_f_volumes = sorted(glob.glob(f"{data_params['data_dir']}/test/volume_f/**.nii.gz"))
    # # test_in_volumes = sorted(glob.glob(f"{data_params['data_dir']}/test/volume_in/**.nii.gz"))
    # test_labels = sorted(glob.glob(f"{data_params['data_dir']}/test/label/**.nii.gz"))

    # ds_train = MRIDataset(train_volumes, train_labels, transforms=transform, thickSlice=None, water_vols=train_w_volumes, fat_vols=None, in_vols = None, orientation='AXI', is_train=True)
    # train_loader = torch.utils.data.DataLoader(ds_train, batch_size=train_params['train_batch_size'], shuffle=True,
    #                                            num_workers=4, pin_memory=True)

    # ds_test = MRIDataset(test_volumes, test_labels, transforms=None, thickSlice=None, water_vols=test_w_volumes, fat_vols=None, in_vols = None, orientation='AXI', is_train=False)
    # val_loader = torch.utils.data.DataLoader(ds_test, batch_size=train_params['val_batch_size'], shuffle=False,
    #                                          num_workers=4, pin_memory=True)

    if train_params['use_pre_trained']:
        model = torch.load(train_params['pre_trained_path'])
    else:
        net_params_ = net_params.copy()
        if net_params['type'] == 'quicknat':
            model = mclass(net_params)
            empty_model = mclass(net_params_)

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

    solver.train(None, None, (train_volumes, train_w_volumes, train_labels, (orientation, train_params['train_batch_size'], train_params['val_batch_size'])))
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
