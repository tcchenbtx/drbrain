import os
from utils import process_nii

# path
base_path = os.path.abspath(os.path.dirname(__file__))
base_path = os.path.join(base_path, "..")
raw_data_path = os.path.join(base_path, "raw_data")
code_path = os.path.join(base_path, "code")
log_path = os.path.join(code_path, "log")

# path to where cropped nii stores
preprocess_data_path = os.path.join(base_path, "preprocess_data")
AD_crop_path = os.path.join(preprocess_data_path, "crop_figure", "AD")
MCI_crop_path = os.path.join(preprocess_data_path, "crop_figure", "MCI")
Normal_crop_path = os.path.join(preprocess_data_path, "crop_figure", "Normal")
PD_crop_path = os.path.join(preprocess_data_path, "crop_figure", "PD")

# path to the Makefile
make_file_path = os.path.join(code_path, "Makefile")

# path to where to save the skull strip nii
AD_skullstrip_path = os.path.join(preprocess_data_path, "skull_strip", "AD")
MCI_skullstrip_path = os.path.join(preprocess_data_path, "skull_strip", "MCI")
Normal_skullstrip_path = os.path.join(preprocess_data_path, "skull_strip", "Normal")
PD_skullstrip_path = os.path.join(preprocess_data_path, "skull_strip", "PD")


dir_check = [preprocess_data_path, AD_skullstrip_path, MCI_skullstrip_path, Normal_skullstrip_path,PD_skullstrip_path ]
for i in dir_check:
    if not os.path.exists(i):
        os.makedirs(i)

# work on PD first
#process_nii.skull_strip("AD1", AD_crop_path, AD_skullstrip_path, make_file_path, log_path)
#process_nii.skull_strip("MCI1", MCI_crop_path, MCI_skullstrip_path, make_file_path, log_path)
process_nii.skull_strip("Normal", Normal_crop_path, Normal_skullstrip_path, make_file_path, log_path)
# process_nii.skull_strip("PD1", PD_crop_path, PD_skullstrip_path, make_file_path, log_path)



