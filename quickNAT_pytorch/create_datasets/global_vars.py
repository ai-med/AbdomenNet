
dset = 'UKB'
DATASET = dset.upper()
dataset = dset.lower()
DEFAULT_FILE_TYPE = 'nifti'
TARGET_FILE_TYPE = 'nifti'
DEFAULT_ORIENTATION = 'RAS'
TARGET_RESOLUTION = [2,2,3]
DEFAULT_VIEW = ['Saggital', 'Coronal', 'Axial']
DEFAULT_REFERENCE_VIEW = 'Sagittal'
OPTIMIZATION = 'N4'  # Intensity, Min-Max, Fat-Water-Swap
IS_CROPPING = True
DEFAULT_OUTPUT_PATH = '/mnt/nas/Abhijit/Jyotirmay/abdominal_segmentation/temp2'
DEFAULT_LINSPACE = 30


FILE_TO_LABEL_MAP =  {'BACKGROUND': ['background'],
                      'LIVER': ['liver'], 
                      'SPLEEN': ['spleen', 'spl'],
                      'KIDNEY(RIGHT)':['kidney_r', 'kidney (right)', 'kidney (right).nrrd', 'kidney (rechts)'],
                      'KIDNEY(LEFT)':['kidney_l', 'kidney (left)', 'kidney (left).nrrd', 'kidney (links)'], 
#                       'ADRENAL': ['adrenal', 'adremal', 'adremalgland(right)', 'adrenalgland(right)','adrenalgalnd(right)', 'adremalgland(left)', 'adrenalgland(left)', 'adrenalgalnd(left)', 'adrenalgland(links)'],
                      'ADRENAL(RIGHT)':['adremalgland(right)', 'adrenalgland(right)','adrenalgalnd(right)'], 
                      'ADRENAL(LEFT)': ['adremalgland(left)', 'adrenalgland(left)', 'adrenalgalnd(left)', 'adrenalgland(links)'],
                      'PANCREAS': ['pancreas'],
                      'GALLBLADDER': ['gallbladder', 'Gallblader']}

LABEL_EXTENSION_FOR_OVERLAP_REMOVAL = 100 # len(FILE_TO_LABEL_MAP) + 1 # Minimum value
HIST_MATCHING_VOL_PATH = '/mnt/nas/Data_WholeBody/NAKO/NAKO_200/MRI/100129/100129_3D_GRE_TRA_opp/3D_GRE_TRA_opp_3D_GRE_TRA_1.nii.gz' # None

# 'SUBCUTANEOUS':['subcutaneous', 'subcutan'], 'THYROIDGLAND':['thyroid']

volume_txt_file = f'datasets/{dataset}/volumes.txt' #volumes.txt'
if DATASET == 'KORA':
    data_dir = "/mnt/nas/Data_WholeBody/KORA/KORA_all/KORA_Nifti"
    SAMPLING = True
elif DATASET == 'NAKO':
    data_dir = f"/mnt/nas/Data_WholeBody/NAKO/NAKO_200/MRI"
    SAMPLING = True
elif DATASET == 'UKB':
    data_dir = "/mnt/nas/Data_WholeBody/UKBiobank/body/body_nifti"
    # data_dir = "/mnt/nas/Users/Sebastian/UKB_nifty/"
    SAMPLING = True
else:
    raise Exception('NO DATA DIRECTORY FOUND!!!')

label_dir = f'datasets/lablmaps/{DATASET}'

n4_corrected_data_dir = f"{DEFAULT_OUTPUT_PATH}/{DATASET}/n4_corrected_2"

processed_path = f'{DEFAULT_OUTPUT_PATH}/{DATASET}/' if SAMPLING else f'{DEFAULT_OUTPUT_PATH}/{DATASET}/unsampled'
processed_dir = f'{DEFAULT_OUTPUT_PATH}/{DATASET}/processed' if SAMPLING else f'{DEFAULT_OUTPUT_PATH}/{DATASET}/processed/unsampled'

one_time_n4_optimization = True
vol_to_check_list = None