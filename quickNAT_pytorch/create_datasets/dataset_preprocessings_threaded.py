# Change the dataset in global_vars.py to UKB, KORA or NAKO as per the run.
# 
from global_vars import *
from commons import *

import glob 
import os
import time
import random
import threading
import multiprocessing
import pandas as pd
import queue

from threading import Thread
num_fetch_threads = 10

queues = [queue.Queue() for _ in range(num_fetch_threads)]

one_time_n4_optimization = True
vol_to_check_list = None #['1942395_20201_2_0'] #['']
exclude = [] #['1004985_20201_2_0', '100006', '100008']

currently_running = []

def downloadEnclosures(i, q):
    """This is the worker thread function.
    It processes items in the queue one after
    another.  These daemon threads go into an
    infinite loop, and only exit when
    the main thread ends.
    """
    while not q.empty():
        print('size of the queue:', q.qsize())
        print('%s: Looking for the next enclosure' % i)
        fp, idx = q.get()
        """ idx: Volume serial number
        fp: File Paths"""
        print('%s: Started:' % i, idx, fp)
        stitching(fp, idx)
        q.task_done()
    print(f'All finished for thread: {i}')
        
def stitching(file_paths, threadid):
    # STITCHING VOL PARTS HERE
    print(f'####################### Process started for threadid: {threadid} ##############################')
    for vol in file_paths.keys():
        try:
            print(f'{vol}:: process started...')
            if vol_to_check_list is not None and vol not in vol_to_check_list:
                continue
            create_if_not(f'{n4_corrected_data_dir}/vol/{vol}')
            file_paths[vol]['ONE'] = {}
            for modality_key in file_paths[vol]['VOLUME_PATHS'].keys():
                print(f"{vol}:: processing {modality_key}")
                orig_modal_key = modality_key
                if one_time_n4_optimization:
                    # pass
                    vol_parts = [read_ras(file) for file in file_paths[vol]['VOLUME_PATHS'][modality_key]]
                else:
                    if modality_key == 'OPP':
                        vol_parts = [read_ras(data_dict['OPP_CORRECTED']) for data_dict in file_paths[vol]['N4_1']]
                        modality_key = modality_key+'_n4_corrected'
                    else:
                        vol_parts = [read_ras(file) for file in file_paths[vol]['VOLUME_PATHS'][modality_key]]

                ras_stitched = multi_vol_stitching(vol_parts, sampling=SAMPLING)
                save_volume(ras_stitched, f'{n4_corrected_data_dir}/vol/{vol}/{modality_key}_ras_stitched', np.int16)
                file_paths[vol]['ONE'][f'{orig_modal_key}'] = f'{n4_corrected_data_dir}/vol/{vol}/{modality_key}_ras_stitched.nii.gz'
                
            file_paths = n4_process(file_paths)
        except Exception as e:
            print('ERROR:', e)
            continue
    return file_paths
            
def n4_process(file_paths):
    # RESCALING INTENSITIES OF IN PHASE STITCHED VOLUME FROM THE file_paths,
    # DO N4 NORMALIZATION --> DO CROPPING --> SAVE OUTPUT
    n4_dict = {}
    ukb_indices = [29, 221, 10, 186, 72, 184]
    kora_indices = [32, 224, 2, 178, 69, 181]
    nako_indices = [20, 212, 0, 176, 157, 269]
    if DATASET=='KORA':
        indices = kora_indices
    elif DATASET=='NAKO':
        indices = nako_indices
    elif DATASET == 'UKB':
        indices = ukb_indices
    else:
        raise Exception('Invalid dataset indices!')
    for vol in file_paths.keys():
        try:
            if vol_to_check_list is not None and vol not in vol_to_check_list:
                continue
            print(f'{vol}:: n4 bias-field correction started')
            if vol_to_check_list is not None and vol not in vol_to_check_list:
                continue
            n4_dict[vol] = {}
            in_stitched_file_path, in_stitched_img = file_paths[vol]['ONE']['IN'], read_ras(file_paths[vol]['ONE']['IN'])
            n4_dict[vol]['N4_2'] = rescale(in_stitched_img, vol, in_stitched_file_path)
            
            in_file = n4_dict[vol]['N4_2']['SCALED']
            if all_n4_process:
                w_file = file_paths[vol]['ONE']['W']
                w_of = w_file.split('/')[-1].split('.')[0]
                w_outputfile = f'{n4_corrected_data_dir}/vol/{vol}/{w_of}_n4_corrected_sitk.nii.gz'
                SITK_N4_normalization(in_file, w_file, w_outputfile)
                crop([w_file], indices, resize(nb.load(w_file)))
                crop([w_outputfile], indices, resize(nb.load(w_outputfile)))
                n4_dict[vol]['N4_2']['W_CORRECTED'] = w_outputfile

                f_file = file_paths[vol]['ONE']['F']
                f_of = f_file.split('/')[-1].split('.')[0]
                f_outputfile = f'{n4_corrected_data_dir}/vol/{vol}/{f_of}_n4_corrected_sitk.nii.gz'
                SITK_N4_normalization(in_file, f_file, f_outputfile)
                crop([f_file], indices, resize(nb.load(f_file)))
                crop([f_outputfile], indices, resize(nb.load(f_outputfile)))
                n4_dict[vol]['N4_2']['F_CORRECTED'] = f_outputfile

                inin_file = file_paths[vol]['ONE']['IN']
                in_of = inin_file.split('/')[-1].split('.')[0]
                in_outputfile = f'{n4_corrected_data_dir}/vol/{vol}/{in_of}_n4_corrected_sitk.nii.gz'
                SITK_N4_normalization(in_file, inin_file, in_outputfile)
                crop([inin_file], indices, resize(nb.load(inin_file)))
                crop([in_outputfile], indices, resize(nb.load(in_outputfile)))
                n4_dict[vol]['N4_2']['IN_CORRECTED'] = in_outputfile

            opp_file = file_paths[vol]['ONE']['OPP']
            new_filename = opp_file.split('/')[-1].split('.')[0]
            output_file = f'{n4_corrected_data_dir}/vol/{vol}/{new_filename}_n4_corrected_sitk.nii.gz'
            SITK_N4_normalization(in_file, opp_file, output_file)
            n4_dict[vol]['N4_2']['OPP_CORRECTED'] = output_file
            crop([opp_file], indices, resize(nb.load(opp_file)))
            crop([output_file], indices, resize(nb.load(output_file)))
            file_paths[vol]['N4_2'] = n4_dict[vol]['N4_2']
        except Exception as e:
            print('ERROR:',e)
            continue
    # Going for final process:
    if DO_VOL_LABEL_ALIGNMENT:
        vol_label_alignment(file_paths)
    
    print(f'Process Complete')
    
def vol_label_alignment(file_paths):
    if DATASET=='KORA':
        kora_vol_label_allignment(file_paths)
    elif DATASET=='NAKO':
        nako_vol_label_allignment(file_paths)
    elif DATASET=='UKB':
        ukb_vol_label_allignment(file_paths)
    else:
        raise Exception('Not a valid dataset for vol-label alignment!')
    
def load_dataset_file_paths(load_from_txt_file=True):
    # LOAD FROM TEXT FILES WITH LIST OF VOLUME ID TO BE PROCESSED ELSE PROCESS ALL IN "data_dir" PATH FROM global_vars.py.
    volumes_to_use = []
    jobs = []
    if load_from_txt_file:
        with open(VOLUME_TXT_FILE) as file_handle:
                volumes_to_use = file_handle.read().splitlines()
    else:
        volumes_to_use = [name for name in os.listdir(data_dir)]
    
    print(f"Total number of volumes to be processed: {len(volumes_to_use)}")
    file_paths = {}
    ii = 0
    for idx, vol in enumerate(volumes_to_use):
        if (vol_to_check_list is not None and vol not in vol_to_check_list) or (vol == "") or (vol in exclude):
            continue
        if (len(glob.glob(f'{DEFAULT_OUTPUT_PATH}/{DATASET}/n4_corrected_2/vol/{vol}/**')) in [17]):
            print('Already Processed, skipping', vol, idx, ii)
            continue
        print(idx, ii)
        if DATASET=='KORA':
            # KORA
            opp_paths = glob.glob(f'{data_dir}/{vol}/**opp_[0-9]**.nii.gz')
            in_paths = glob.glob(f'{data_dir}/{vol}/**in_[0-9]**.nii.gz')
            f_paths = glob.glob(f'{data_dir}/{vol}/**F_[0-9]**.nii.gz')
            w_paths = glob.glob(f'{data_dir}/{vol}/**W_[0-9]**.nii.gz')
            labelmap_paths = glob.glob(f'{label_dir}/{vol}/**')
        elif DATASET=='NAKO':
             # NAKO
            opp_paths = glob.glob(f'{data_dir}/{vol}/{vol}_3D_GRE_TRA_opp/**.nii.gz') # **_2**
            in_paths = glob.glob(f'{data_dir}/{vol}/{vol}_3D_GRE_TRA_in/**.nii.gz')
            f_paths = glob.glob(f'{data_dir}/{vol}/{vol}_3D_GRE_TRA_F/**.nii.gz')
            w_paths = glob.glob(f'{data_dir}/{vol}/{vol}_3D_GRE_TRA_W/**.nii.gz')
            labelmap_paths = glob.glob(f'{label_dir}/{vol}/**')
        elif DATASET=='UKB':
            # UKB
            opp_paths = glob.glob(f'{data_dir}/{vol}/**opp**_[17s, 17sa,17sb]**.nii.gz')
            in_paths = glob.glob(f'{data_dir}/{vol}/**in**_[17s,17sa,17sb]**.nii.gz')
            f_paths = glob.glob(f'{data_dir}/{vol}/**F**_[17s, 17sa,17sb]**.nii.gz')
            w_paths = glob.glob(f'{data_dir}/{vol}/**W**_[17s, 17sa,17sb]**.nii.gz')
            labelmap_paths = glob.glob(f'{label_dir}/{vol}/**')
        else:
            raise Exception('Not a valid dataset, please change appropriately on global_vars.py! Valid options are KORA, NAKO or UKB.')

        vol_madals_paths = dict(
        OPP=opp_paths,
        IN=in_paths,
        F=f_paths,
        W=w_paths
        )
        file_paths[str(vol)]=dict(
            VOLUME_PATHS=vol_madals_paths,
            LABEL_PATHS=labelmap_paths,
        )
        fp = {str(vol): dict(
            VOLUME_PATHS=vol_madals_paths,
            LABEL_PATHS=labelmap_paths,
        )}
        try:
            ii+=1
            queues[int(idx%num_fetch_threads)].put((fp, idx))
        except Exception as e:
            print("ERRROR $$$$$$$$$$$$$$$$$$$$$$$$$$ : ", e)
    print("List processing complete. total:", ii)
    
    return file_paths

file_paths = load_dataset_file_paths(False)
for i in range(num_fetch_threads):
    time.sleep(2)
    worker = multiprocessing.Process(target=downloadEnclosures, args=(i, queues[i]))
    worker.start()

print('*** Main thread waiting')
_ = [qu.join() for qu in queues]
print('*** Done')