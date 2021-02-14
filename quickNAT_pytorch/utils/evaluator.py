import os
from quicknat import QuickNat
# from quick_oct import QuickOct
import nibabel as nb
import numpy as np

import logging
import torch
import utils.common_utils as common_utils
import utils.data_utils as du
from .preprocessor import remove_black_3channels
import torch.nn.functional as F

log = logging.getLogger(__name__)

to_axis_dict = {
        "AXI": 2,
        "COR": 1,
        "SAG": 0
    }

def dice_confusion_matrix(vol_output, ground_truth, classes, no_samples=10, mode='train'):
    dice_cm = torch.zeros(len(classes), len(classes))

    print('dice cm ', dice_cm.shape)
    if mode == 'train':
        samples = np.random.choice(len(vol_output), no_samples)
        vol_output, ground_truth = vol_output[samples], ground_truth[samples]
    for i, c in enumerate(classes):
        GT = (ground_truth == c).float()
        for j, c in enumerate(classes):
            Pred = (vol_output == c).float()
            inter = torch.sum(torch.mul(GT, Pred))
            union = torch.sum(GT) + torch.sum(Pred) + 0.0001
            dice_cm[i, j] = 2 * torch.div(inter, union)
    avg_dice = torch.mean(torch.diagflat(dice_cm))
    return avg_dice, dice_cm


def dice_score_perclass(vol_output, ground_truth, classes, no_samples=10, mode='train'):
    dice_perclass = torch.zeros(len(classes))
    if mode == 'train':
        samples = np.random.choice(len(vol_output), no_samples)
        vol_output, ground_truth = vol_output[samples], ground_truth[samples]
    for i, c in enumerate(classes):
        GT = (ground_truth == c).float()
        Pred = (vol_output == c).float()
        inter = torch.sum(torch.mul(GT, Pred))
        union = torch.sum(GT) + torch.sum(Pred) + 0.0001
        dice_perclass[i] = (2 * torch.div(inter, union))
    return dice_perclass

def evaluate_dice_score(model_path, num_classes, data_dir, label_dir, volumes_txt_file, orientation,
                        prediction_path, device=0, logWriter=None, mode='eval', multi_channel=False, use_2channel=False, thick_ch=False):
    log.info("**Starting evaluation. Please check tensorboard for plots if a logWriter is provided in arguments**")
    batch_size = 15

    with open(volumes_txt_file) as file_handle:
        volumes_to_use = file_handle.read().splitlines()
    if multi_channel or use_2channel:
        file_paths = du.load_file_paths_3channel(data_dir, label_dir, volumes_txt_file)
    else:
        file_paths = du.load_file_paths(data_dir, label_dir, volumes_txt_file)

    cuda_available = torch.cuda.is_available()
    # First, are we attempting to run on a GPU?
    if type(device) == int:
        # if CUDA available, follow through, else warn and fallback to CPU
        if cuda_available:
            model = torch.load(model_path)
            torch.cuda.empty_cache()
            model.cuda(device)
        else:
            log.warning(
                'CUDA is not available, trying with CPU.' + \
                'This can take much longer (> 1 hour). Cancel and ' + \
                'investigate if this behavior is not desired.'
            )
            # switch device to 'cpu'
            device = 'cpu'
    # If device is 'cpu' or CUDA not available
    if (type(device) == str) or not cuda_available:
        model = torch.load(
            model_path,
            map_location=torch.device(device)
        )

    model.eval()

    common_utils.create_if_not(prediction_path)
    volume_dice_score_list = []
    log.info("Evaluating now...")

    with torch.no_grad():
        for vol_idx, file_path in enumerate(file_paths):
            if multi_channel:
                img, label, water, inv = nb.load(file_path[0]), nb.load(file_path[1]), nb.load(file_path[2]), nb.load(file_path[4])
                volume, labelmap, water, inv, class_weights, weights, header, affine = img.get_fdata(), label.get_fdata(), water.get_fdata(), inv.get_fdata(), None, None, img.header, img.affine

                volume = np.rollaxis(volume, to_axis_dict[orientation], 0)
                labelmap = np.rollaxis(labelmap, to_axis_dict[orientation], 0)
                water = np.rollaxis(water, to_axis_dict[orientation], 0)
                # fat = np.rollaxis(fat, to_axis_dict[orientation], 0)
                inv = np.rollaxis(inv, to_axis_dict[orientation], 0)

                template = np.zeros_like(labelmap)
                volume, _, water, labelmap, inv, S, E = remove_black_3channels(volume, None, water, labelmap, inv, return_indices=True)

                thick_volume = []
                for w, v, ij in zip(water, volume, inv):
                    thick_volume.append(np.stack([w, v, ij], axis=0))
                volume = np.array(thick_volume)

            elif use_2channel:
                img, label, water = nb.load(file_path[0]), nb.load(file_path[1]), nb.load(file_path[2])
                volume, labelmap, water, class_weights, weights, header, affine = img.get_fdata(), label.get_fdata(), water.get_fdata(), None, None, img.header, img.affine

                volume = np.rollaxis(volume, to_axis_dict[orientation], 0)
                labelmap = np.rollaxis(labelmap, to_axis_dict[orientation], 0)
                water = np.rollaxis(water, to_axis_dict[orientation], 0)
                template = np.zeros_like(labelmap)
                volume, _, water, labelmap, _, S, E = remove_black_3channels(volume, None, water, labelmap, None, return_indices=True)

                print(volume.shape, water.shape, labelmap.shape)
                thick_volume = []
                for v, w in zip(volume, water):
                    thick_volume.append(np.stack([w, v], axis=0))
                volume = np.array(thick_volume)
            else:
                img, label = nb.load(file_path[0]), nb.load(file_path[1])
                volume, labelmap, class_weights, weights, header, affine = img.get_fdata(), label.get_fdata(), None, None, img.header, img.affine
                volume = np.rollaxis(volume, to_axis_dict[orientation], 0)
                labelmap = np.rollaxis(labelmap, to_axis_dict[orientation], 0)
                template = np.zeros_like(labelmap)
                volume, _, _, labelmap, _, S, E = remove_black_3channels(volume,None, None, labelmap, None, return_indices=True)
                print(volume.shape, labelmap.shape)
            volume = volume if len(volume.shape) == 4 else volume[:, np.newaxis, :, :]
            volume, labelmap = torch.tensor(volume).type(torch.FloatTensor), torch.tensor(labelmap).type(torch.LongTensor)

            volume_prediction = []
            for i in range(0, len(volume), batch_size):
                if multi_channel or use_2channel:
                    batch_x, batch_y = volume[i: i + batch_size], labelmap[i:i + batch_size]
                elif thick_ch:
                    batch_y = labelmap[i:i + batch_size]
                    batch_x = []
                    volume = np.squeeze(volume)
                    for bs in range(batch_size):
                        index = i+bs
                        if index < 2:
                            n1, n2 = index, index
                        else:
                            n1, n2 = index-1, index-2
                        
                        if index >= volume.shape[0]-3:
                            p1, p2 = index, index
                        else:
                            p1, p2 = index+1, index+2

                        batch_x.append(np.stack([volume[n2], volume[n1], volume[index], volume[p1], volume[p2]], axis=0))
                    batch_x = np.array(batch_x)
                    batch_x = torch.tensor(batch_x).type(torch.FloatTensor)
                else:
                    batch_x, batch_y = volume[i: i + batch_size], labelmap[i:i + batch_size]

                if cuda_available and (type(device)==int):
                    batch_x = batch_x.cuda(device)
                out = model(batch_x)
                _, batch_output = torch.max(out, dim=1)
                volume_prediction.append(batch_output)

            volume_prediction = torch.cat(volume_prediction)
            volume_dice_score = dice_score_perclass(volume_prediction, labelmap.cuda(device), np.arange(0, num_classes),
                                                    mode=mode)

            volume_prediction = (volume_prediction.cpu().numpy()).astype('int16')
            print("evaluator here")
            header.set_data_dtype('int16')
            volume_prediction = np.squeeze(volume_prediction)

            template[S:E] = volume_prediction
            volume_prediction = np.rollaxis(template, 0, to_axis_dict[orientation]+1)

            nifti_img = nb.Nifti1Image(volume_prediction, affine, header=header)
            nb.save(nifti_img, os.path.join(prediction_path, volumes_to_use[vol_idx] + str('_new.nii.gz')))
            if logWriter:
                logWriter.plot_dice_score('val', 'eval_dice_score', volume_dice_score, volumes_to_use[vol_idx],
                                          np.arange(0, num_classes), num_classes)

            volume_dice_score = volume_dice_score.cpu().numpy()
            volume_dice_score_list.append(volume_dice_score)
            log.info(volume_dice_score, np.mean(volume_dice_score))
        dice_score_arr = np.asarray(volume_dice_score_list)
        avg_dice_score = np.mean(dice_score_arr)
        avg_dice_score_wo_bg = np.mean(dice_score_arr[:, 1:])
        log.info("Mean of dice score : " + str(avg_dice_score))
        print('Mean dice score: ', avg_dice_score)
        print('Mean dice score without background: ', avg_dice_score_wo_bg)
        print('all dice scores: ', dice_score_arr)
        print('class wise mean dice scores: ', np.mean(dice_score_arr, axis=0))
        class_dist = [dice_score_arr[:, c] for c in range(num_classes)]

        if logWriter:
            logWriter.plot_eval_box_plot('eval_dice_score_box_plot', class_dist, 'Box plot Dice Score')
    log.info("DONE")

    return avg_dice_score, class_dist


def _segment_vol(file_path, model, orientation, batch_size, cuda_available, device, multi_channel=False, use_2channel=False, remap_config=None):
    
    if multi_channel:
        pass
    elif use_2channel:
        img, label, water = nb.load(file_path[0]), nb.load(file_path[1]), nb.load(file_path[2])
        volume, labelmap, water, class_weights, weights, header, affine = img.get_fdata(), label.get_fdata(), water.get_fdata(), None, None, img.header, img.affine

        volume = np.rollaxis(volume, to_axis_dict[orientation], 0)
        label = np.rollaxis(labelmap, to_axis_dict[orientation], 0)
        water = np.rollaxis(water, to_axis_dict[orientation], 0)

        thick_volume = []
        for v, w in zip(volume, water):
            thick_volume.append(np.stack([w, v], axis=0))
        volume = thick_volume

        volume = torch.tensor(volume).type(torch.FloatTensor)
    else:
        img, labelmap = nb.load(file_path[0]), nb.load(file_path[1])
        volume, labelmap = img.get_fdata(), labelmap.get_fdata()

        volume = np.rollaxis(volume, to_axis_dict[orientation], 0)
        label = np.rollaxis(labelmap, to_axis_dict[orientation], 0)

    volume = volume if len(volume.shape) == 4 else volume[:, np.newaxis, :, :]
    volume = torch.tensor(volume).type(torch.FloatTensor)

    volume_pred = []
    for i in range(0, len(volume), batch_size):
        if multi_channel:
            batch_x = torch.cat((volume[i: i + batch_size], fat[i: i + batch_size], water[i: i + batch_size]),
                                         dim=1)
        elif use_2channel:
            batch_x = volume[i: i + batch_size]
        else:
            batch_x = volume[i: i + batch_size]

        if cuda_available and (type(device) == int):
            batch_x = batch_x.cuda(device)
        out = model(batch_x)
        print(out.shape)
        volume_pred.append(out)

    volume_pred = torch.cat(volume_pred)
    _, volume_prediction = torch.max(volume_pred, dim=1)

    volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
    volume_prediction = np.squeeze(volume_prediction)

    if orientation == "AXI":
        volume_prediction = volume_prediction.transpose((1, 2, 0))
        reference_label = label.transpose((1, 2, 0))
        volume_pred = volume_pred.permute((2, 1, 3, 0))
    elif orientation == "COR":
        volume_prediction = volume_prediction.transpose((1, 0, 2))
        reference_label = label.transpose((1, 0, 2))
        volume_pred = volume_pred.permute((2, 1, 0, 3))
    else:
        reference_label = label

    return volume_pred, (label, reference_label), volume_prediction, header

def evaluate3view(coronal_model_path, axial_model_path, sagittal_model_path, volumes_txt_file, data_dir, label_dir,
                  device, prediction_path,
                  batch_size,
                  label_names, label_list, exit_on_error=False, multi_channel=False,use_2channel=False):
    log.info("**Starting evaluation**")

    with open(volumes_txt_file) as file_handle:
        volumes_to_use = file_handle.read().splitlines()

    if multi_channel or use_2channel:
        file_paths = du.load_file_paths_3channel(data_dir, label_dir, volumes_txt_file)
    else:
        file_paths = du.load_file_paths(data_dir, label_dir, volumes_txt_file)

    cuda_available = torch.cuda.is_available()
    if type(device) == int:
        # if CUDA available, follow through, else warn and fallback to CPU
        if cuda_available:
            model1 = torch.load(coronal_model_path)
            model2 = torch.load(axial_model_path)
            model3 = torch.load(sagittal_model_path)

            torch.cuda.empty_cache()
            model1.cuda(device)
            model2.cuda(device)
            model3.cuda(device)
        else:
            log.warning(
                'CUDA is not available, trying with CPU.' + \
                'This can take much longer (> 1 hour). Cancel and ' + \
                'investigate if this behavior is not desired.'
            )

    if (type(device) == str) or not cuda_available:
        model1 = torch.load(
            coronal_model_path,
            map_location=torch.device(device)
        )
        model2 = torch.load(
            axial_model_path,
            map_location=torch.device(device)
        )
        model3 = torch.load(
            axial_model_path,
            map_location=torch.device(device)
        )

    model1.eval()
    model2.eval()
    model3.eval()

    common_utils.create_if_not(prediction_path)
    log.info("Evaluating now...")

    print(file_paths)

    with torch.no_grad():
        volume_dict_list = []
        cvs_dict_list = []
        iou_dict_list = []
        all_dice_scores = np.zeros((9))
        for vol_idx, file_path in enumerate(file_paths):

            volume_prediction_cor, (label, reference_label), _, header = _segment_vol(file_path, model1, "COR",
                                                                                        batch_size, cuda_available, device, multi_channel, use_2channel)
            print('segment cor')
            volume_prediction_axi, (label, reference_label), _, header = _segment_vol(file_path, model2, "AXI", batch_size,
                                                                        cuda_available,
                                                                        device, multi_channel, use_2channel)
            print('segment axi')
            volume_prediction_sag, (label, reference_label), _, header = _segment_vol(file_path, model3, "SAG", batch_size,
                                                                        cuda_available,
                                                                        device, multi_channel, use_2channel)
            print('segment sag')

            volume_prediction_axi = F.softmax(volume_prediction_axi, dim=1)
            volume_prediction_cor = F.softmax(volume_prediction_cor, dim=1)
            volume_prediction_sag = F.softmax(volume_prediction_sag, dim=1)

            _, volume_prediction = torch.max(volume_prediction_axi + volume_prediction_sag + volume_prediction_cor,
                                                dim=1)

            volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')

            reference_label = torch.from_numpy(reference_label).cuda(device)
            volume_dice_score = dice_score_perclass(torch.from_numpy(volume_prediction).cuda(device), reference_label,
                                                    label_list, mode='eval')
            print(volume_dice_score)
            all_dice_scores += volume_dice_score.cpu().numpy()

            volume_prediction = np.squeeze(volume_prediction)
            volume_prediction = volume_prediction.astype('int')

            Mat = header.get_best_affine()

            nifti_img = nb.MGHImage(np.squeeze(volume_prediction), Mat, header=header)

            log.info("Processed: " + volumes_to_use[vol_idx] + " " + str(vol_idx + 1) + " out of " + str(
                len(file_paths)))
            ax = axial_model_path.split('/')[-1].split('.')[0]
            co = coronal_model_path.split('/')[-1].split('.')[0]
            sa = sagittal_model_path.split('/')[-1].split('.')[0]
            common_utils.create_if_not(f'{prediction_path}/{ax}_{co}_{sa}')
            nb.save(nifti_img, os.path.join(f'{prediction_path}/{ax}_{co}_{sa}', volumes_to_use[vol_idx] + str('.nii.gz')))

            del volume_prediction, volume_prediction_axi, volume_dice_score, volume_prediction_cor, volume_prediction_sag

        all_dice_scores /= len(file_paths)
        print('avg dice scores: ', all_dice_scores)
        print('mean dice: ', np.mean(all_dice_scores))
        print('mean dice without background: ', np.mean(all_dice_scores[1:]))

    log.info("DONE")
