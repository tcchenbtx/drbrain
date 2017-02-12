import os
from utils import process_nii
from utils import img_check

# path
base_path = os.path.abspath(os.path.dirname(__file__))
base_path = os.path.join(base_path, "..")
raw_data_path = os.path.join(base_path, "raw_data")
code_path = os.path.join(base_path, "code")
log_path = os.path.join(code_path, "log")
preprocess_data_path = os.path.join(base_path, "preprocess_data")

# path to the Makefile
make_file_path = os.path.join(code_path, "Makefile")

# path to where to save the post-flirt nii
AD_afterflirt_path = os.path.join(preprocess_data_path, "after_flirt", "AD")
Normal_afterflirt_path = os.path.join(preprocess_data_path, "after_flirt", "Normal")
PD_afterflirt_path = os.path.join(preprocess_data_path, "after_flirt", "PD")

# get nii paths for all data
AD_nii_files = process_nii.get_nii(AD_afterflirt_path)
Normal_nii_files = process_nii.get_nii(Normal_afterflirt_path)
PD_nii_files = process_nii.get_nii(PD_afterflirt_path)

# too many normal, so split into two parts
Normal_nii_first = Normal_nii_files[:737]
Normal_nii_second = Normal_nii_files[737:]

# create global figures to check images
img_check.check_all(AD_nii_files, "Global_AD_3")
img_check.check_all(Normal_nii_first, "Global_Normal_1")
img_check.check_all(Normal_nii_second, "Global_Normal_2")
img_check.check_all(PD_nii_files, "Global_PD")