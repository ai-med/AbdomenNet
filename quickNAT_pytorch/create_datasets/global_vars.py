
def get_globals(dset):
    pass

dset = 'UKB'
DATASET = dset.upper() # 'NAKO', 'UKB', 'KORANAKOUKB'
dataset = dset.lower()
DEFAULT_FILE_TYPE = 'nifti'  # 'nerd'
TARGET_FILE_TYPE = 'nifti'
DEFAULT_ORIENTATION = 'RAS'
TARGET_RESOLUTION = [2,2,3]
DEFAULT_VIEW = ['Saggital', 'Coronal', 'Axial']
DEFAULT_REFERENCE_VIEW = 'Sagittal'
OPTIMIZATION = 'N4'  # Intensity, Min-Max, Fat-Water-Swap
IS_CROPPING = True
DEFAULT_WORLD_COORDS = [500, 500, 1000]
DEFAULT_OUTPUT_PATH = './temp'
DEFAULT_LINSPACE = 30

FILE_TO_LABEL_MAP =  {'BACKGROUND': ['background'],'LIVER': ['liver'], 'SPLEEN': ['spleen', 'spl'],'KIDNEY(RIGHT)':['kidney_r', 'kidney (right)'],
                      'KIDNEY(LEFT)':['kidney_l', 'kidney (left)'], 'ADRENALGLAND':['adrenal', 'adremal'], 'PANCREAS': ['pancreas'],
                      'GALLBLADDER': ['gallbladder', 'Gallblader']}

# 'SUBCUTANEOUS':['subcutaneous', 'subcutan'], 'THYROIDGLAND':['thyroid']

volume_txt_file = f'datasets/{dataset}/volumes.txt'
if DATASET == 'KORA':
    data_dir = "/home/abhijit/nas_drive/Data_WholeBody/KORA/KORA_all/KORA_Nifti"
elif DATASET == 'NAKO':
    data_dir = f"/home/abhijit/nas_drive/Data_WholeBody/NAKO/NAKO_200/MRI"
elif DATASET == 'UKB':
    data_dir = "/home/abhijit/nas_drive/Data_WholeBody/UKBiobank/body/body_nifti"
else:
    raise Exception('NO DATA DIRECTORY FOUND!!!')

label_dir = f'datasets/lablmaps/{DATASET}'

n4_corrected_data_dir = f"temp/{DATASET}/n4_corrected"
stictched_data_dir = f"temp/{DATASET}/stitched"
stitched_n4_corrected_data_dir = f"temp/{DATASET}/stitched_n4_corrected"

processed_path = f'temp/{DATASET}/'
processed_dir = f'temp/{DATASET}/processed'

one_time_n4_optimization = True

vol_to_check_list = ['KORA2453666']