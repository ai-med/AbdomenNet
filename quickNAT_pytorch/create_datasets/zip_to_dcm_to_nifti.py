# @Description: A fastest version of extracting and converting ukb zip files to dicom to nifti..
# @Author: Jyotirmay S.
# @Date: 23rd January, 2021
# @Owner: ai-med.de.

import os
import subprocess
import pandas as pd
from zipfile import ZipFile
import time
import random
import threading
import multiprocessing
import queue
import shutil
from threading import Thread

# Set number of threads as the machine can work on.
num_fetch_threads = 40

queues = [queue.Queue() for _ in range(num_fetch_threads)]

# MAIN FUNCTION which does the unzipping to dicom to nifti conversion
def zip_dcm_nii(vol, idx):
        print(f'processing_{idx}:{vol}')
        
        zipPath = f"/mnt/nas/Data_WholeBody/UKBiobank/release4/body/zips/{vol}.zip"
        dicomPath = f'/mnt/nas/Abhijit/Jyotirmay/abdominal_segmentation/ukb_extracts/{vol}'
        outputPath = f'/mnt/nas/Abhijit/Jyotirmay/abdominal_segmentation/ukb_niftis/{vol}'
        
        with ZipFile(zipPath, 'r') as zipObj:
           # Extract all the contents of zip file in different directory
            zipObj.extractall(dicomPath)
            
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
#         Install dcm2niix by apt install dcm2niix
        print(['dcm2niix','-z',' ','-f' ,'%d_%p','-o', outputPath, dicomPath])
        subprocess.run(['dcm2niix','-z','y','-f' ,'%d_%p','-o', outputPath, dicomPath])
        try:
            shutil.rmtree(dicomPath)
        except Exception as e:
            print(f'Error: Removing temp dir [ukb dicom extracts] was not successfull for {dicomPath}')
        print('DONE:', outputPath)
        
def start_thread(threadid, q):
    """This is the worker thread function.
    It processes items in the queue one after
    another.  These daemon threads go into an
    infinite loop, and only exit when
    the main thread ends.
    """
    while not q.empty():
        print('size of the queue:', q.qsize())
        print('%s: Looking for the next enclosure' % threadid)
        volid, idx = q.get()
        print('%s: Started:' % threadid, idx, volid)
        zip_dcm_nii(volid, idx)
        q.task_done()


another_path = '/mnt/nas/Data_WholeBody/UKBiobank/release4/body/zips'
df = pd.read_csv('/mnt/nas/Users/Sebastian/UKB/ukb_selected_diabetes_data_matched.csv')
volumes_to_use = df['Unnamed: 0'].values.tolist()
dirs = [fs.split('.')[0] for fs in os.listdir(another_path)]

# Distribute volume ids into n queues which will serve as data for n threads.
# Processing only volumes present in the dataframe now.
for idx, vol in enumerate(volumes_to_use):
    vol = f"{vol}_20201_2_0"
    if vol in dirs:
        try:
            extracts_path = f'/mnt/nas/Abhijit/Jyotirmay/abdominal_segmentation/ukb_extracts/{vol}'
            nii_path = f'/mnt/nas/Abhijit/Jyotirmay/abdominal_segmentation/ukb_niftis/{vol}'
            if os.path.exists(nii_path):
                print(len(os.listdir(nii_path)))
                if len(os.listdir(nii_path)) == 48:
#                     shutil.rmtree(extracts_path)
                    continue
                
            queues[int(idx%num_fetch_threads)].put((vol, idx))
        except Exception as e:
            print("ERROR $$$$$$$$$$$$$$$$$$$$$$$$$$ : ", e)
            
for threadid in range(num_fetch_threads):
    time.sleep(2)
    worker = multiprocessing.Process(target=start_thread, args=(threadid, queues[threadid]))
    worker.start()

print('*** Main thread waiting')
_ = [qu.join() for qu in queues]
print('*** Done')