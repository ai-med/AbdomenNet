# Change the dataset in global_vars.py to UKB.
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
# from multiprocessing import Queue
from threading import Thread
num_fetch_threads = 50

queues = [queue.Queue() for _ in range(num_fetch_threads)]

one_time_n4_optimization = True
vol_to_check_list = None #['1942395_20201_2_0'] #['']
exclude = ['1004985_20201_2_0']

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
        print('%s: Started:' % i, idx, fp)
        stitching(fp, idx)
        q.task_done()
        
def stitching(file_paths, threadid):
    # STITCHING VOL PARTS HERE
    print(f'####################### Process started for threadid: {threadid} ##############################')
    for vol in file_paths.keys():
        try:
            print(f'{vol}:: started with {vol}...')
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
    # RESCALING INTENSITIES OF STITCHED VOLUME ABOVE 0
    n4_dict = {}
    ukb_indices = [29, 221, 10, 186, 72, 184]
    all_n4_process = True
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
            opp_file = file_paths[vol]['ONE']['OPP']
            if all_n4_process:
                w_file = file_paths[vol]['ONE']['W']
                w_of = w_file.split('/')[-1].split('.')[0]
                w_outputfile = f'{n4_corrected_data_dir}/vol/{vol}/{w_of}_n4_corrected_sitk.nii.gz'
                SITK_N4_normalization(in_file, w_file, w_outputfile)
                crop([w_file], ukb_indices, resize(nb.load(w_file)))
                crop([w_outputfile], ukb_indices, resize(nb.load(w_outputfile)))
                n4_dict[vol]['N4_2']['W_CORRECTED'] = w_outputfile

                f_file = file_paths[vol]['ONE']['F']
                f_of = f_file.split('/')[-1].split('.')[0]
                f_outputfile = f'{n4_corrected_data_dir}/vol/{vol}/{f_of}_n4_corrected_sitk.nii.gz'
                SITK_N4_normalization(in_file, f_file, f_outputfile)
                crop([f_file], ukb_indices, resize(nb.load(f_file)))
                crop([f_outputfile], ukb_indices, resize(nb.load(f_outputfile)))
                n4_dict[vol]['N4_2']['F_CORRECTED'] = f_outputfile

                inin_file = file_paths[vol]['ONE']['IN']
                in_of = inin_file.split('/')[-1].split('.')[0]
                in_outputfile = f'{n4_corrected_data_dir}/vol/{vol}/{in_of}_n4_corrected_sitk.nii.gz'
                SITK_N4_normalization(in_file, inin_file, in_outputfile)
                crop([inin_file], ukb_indices, resize(nb.load(inin_file)))
                crop([in_outputfile], ukb_indices, resize(nb.load(in_outputfile)))
                n4_dict[vol]['N4_2']['IN_CORRECTED'] = in_outputfile


            new_filename = opp_file.split('/')[-1].split('.')[0]
            output_file = f'{n4_corrected_data_dir}/vol/{vol}/{new_filename}_n4_corrected_sitk.nii.gz'
            SITK_N4_normalization(in_file, opp_file, output_file)
            n4_dict[vol]['N4_2']['OPP_CORRECTED'] = output_file
            crop([opp_file], ukb_indices, resize(nb.load(opp_file)))
            crop([output_file], ukb_indices, resize(nb.load(output_file)))
            file_paths[vol]['N4_2'] = n4_dict[vol]['N4_2']
        except Exception as e:
            print('ERROR:',e)
            continue
    return file_paths
        
def load_ukb_file_paths(load_from_txt_file=True):
    volumes_to_use = []
    jobs = []
    if load_from_txt_file:
        with open(volume_txt_file) as file_handle:
                volumes_to_use = file_handle.read().splitlines()
    else:
        volumes_to_use = [name for name in os.listdir(data_dir)]
        # df = pd.read_csv('/mnt/nas/Users/Sebastian/UKB/ukb_selected_diabetes_data_matched.csv')
        # volumes_to_use = df['Unnamed: 0'].values.tolist()
    
    print(len(volumes_to_use))
    file_paths = {}
    
    for idx, vol in enumerate(volumes_to_use):
        # vol = f"{vol}_20201_2_0"
        if (vol_to_check_list is not None and vol not in vol_to_check_list) or (vol == "") or (vol in exclude):
            continue
        if vol != '1002359_20201_2_0':
            continue
        if (len(glob.glob(f'{DEFAULT_OUTPUT_PATH}/UKB/n4_corrected_2/vol/{vol}/**')) in [9, 17]):
#             print('skipping', vol)
            continue

        if (len(glob.glob(f'/mnt/nas/Abhijit/Jyotirmay/abdominal_segmentation/temp3/UKB/n4_corrected_2/vol/{vol}/**')) in [9, 17]):
            continue
            
        opp_paths = glob.glob(f'{data_dir}/{vol}/**opp**_[17s, 17sa,17sb]**.nii.gz')
        in_paths = glob.glob(f'{data_dir}/{vol}/**in**_[17s,17sa,17sb]**.nii.gz')
        f_paths = glob.glob(f'{data_dir}/{vol}/**F**_[17s, 17sa,17sb]**.nii.gz')
        w_paths = glob.glob(f'{data_dir}/{vol}/**W**_[17s, 17sa,17sb]**.nii.gz')
        
        labelmap_paths = glob.glob(f'{label_dir}/{vol}/**')
        
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
            queues[int(idx%num_fetch_threads)].put((fp, idx))
        except Exception as e:
            print("ERRRORRRRRRR $$$$$$$$$$$$$$$$$$$$$$$$$$ : ", e)
    print("List processing complete.")
    
    return file_paths

file_paths = load_ukb_file_paths(False)
for i in range(num_fetch_threads):
    time.sleep(2)
    worker = multiprocessing.Process(target=downloadEnclosures, args=(i, queues[i]))
    worker.start()

print('*** Main thread waiting')
_ = [qu.join() for qu in queues]
print('*** Done')