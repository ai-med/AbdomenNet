import argparse
import os
import torch
from utils.evaluator import evaluate, evaluate2view, evaluate_dice_score, compute_vol_bulk, evaluate3view
from quicknat import QuickNat
from fastSurferCNN import FastSurferCNN
from settings import Settings
from solver import Solver
from utils.data_utils import get_imdb_dataset, get_imdb_dataset_3channel, get_imdb_dataset_2channel
from utils.log_utils import LogWriter
import logging
import shutil
import glob
import nibabel as nb
import numpy as np
# from create_datasets.commons import MRIDataset
import torch.utils.data as data

torch.set_default_tensor_type('torch.FloatTensor')


class MRIDataset(data.Dataset):
    def __init__(self, X_files, y_files, transforms=None):
        self.X_files = X_files
        self.y_files = y_files
        self.transforms = transforms

        img_array = list()
        label_array = list()
        for vol_f, label_f in zip(self.X_files, self.y_files):
            img, label = nb.load(vol_f), nb.load(label_f)
            img_data = np.array(img.get_fdata())
            label_data = np.array(label.get_fdata())

            # Transforming to Axial Manually.
            img_data = np.rollaxis(img_data, 2, 0)
            label_data = np.rollaxis(label_data, 2, 0)

            img_array.extend(img_data)
            label_array.extend(label_data)
            img.uncache()
            label.uncache()

        X = np.stack(img_array, axis=0) if len(img_array) > 1 else img_array[0]
        y = np.stack(label_array, axis=0) if len(label_array) > 1 else label_array[0]
        self.X = X if len(X.shape) == 4 else X[:, np.newaxis, :, :]
        self.y = y
        print(self.X.shape, self.y.shape)

    def __getitem__(self, index):
        img = torch.from_numpy(self.X[index])
        label = torch.from_numpy(self.y[index])
        return img, label

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

    train_volumes = sorted(glob.glob(f"{data_params['data_dir']}/volume_cropped_train/**.nii.gz"))
    train_labels = sorted(glob.glob(f"{data_params['data_dir']}/label_cropped_train/**.nii.gz"))

    test_volumes = sorted(glob.glob(f"{data_params['data_dir']}/volume_cropped_test/**.nii.gz"))
    test_labels = sorted(glob.glob(f"{data_params['data_dir']}/label_cropped_test/**.nii.gz"))

    ds_train = MRIDataset(train_volumes, train_labels)
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=train_params['train_batch_size'], shuffle=True,
                                               num_workers=4, pin_memory=True)

    ds_test = MRIDataset(test_volumes, test_labels)
    val_loader = torch.utils.data.DataLoader(ds_test, batch_size=train_params['train_batch_size'], shuffle=False,
                                             num_workers=4, pin_memory=True)

    if train_params['use_pre_trained']:
        model = torch.load(train_params['pre_trained_path'])
    else:
        if net_params['type'] == 'quicknat':
            model = QuickNat(net_params)
        elif net_params['type'] == 'fastsurfer':
            model = FastSurferCNN(net_params)

    solver = Solver(model,
                    device=common_params['device'],
                    num_class=net_params['num_class'],
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
                    exp_dir=common_params['exp_dir'])

    solver.train(train_loader, val_loader)
    final_model_path = os.path.join(common_params['save_model_dir'], train_params['final_model_file'])
    model.save(final_model_path)
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
                                                        use_2channel=use_2channel)
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
    parser.add_argument('--setting_path', '-sp', required=False, help='optional path to settings_eval_nako.ini')
    args = parser.parse_args()

    settings = Settings('/home/abhijit/Jyotirmay/abdominal_segmentation/quickNAT_pytorch/settings_merged_jj.ini')
    common_params, data_params, net_params, train_params, eval_params = settings['COMMON'], settings['DATA'], \
                                                                        settings[
                                                                            'NETWORK'], settings['TRAINING'], \
                                                                        settings['EVAL']
    if args.mode == 'train':
        train(train_params, common_params, data_params, net_params)
    elif args.mode == 'eval':
        evaluate(eval_params, net_params, data_params, common_params, train_params)
    elif args.mode == 'eval_bulk':
        logging.basicConfig(filename='error.log')
        if args.setting_path is not None:
            settings_eval = Settings(args.setting_path)
        else:
            settings_eval = Settings('settings_kora.ini')
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
