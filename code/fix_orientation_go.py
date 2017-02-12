import os
from utils import process_nii

# path
base_path = os.path.abspath(os.path.dirname(__file__))
base_path = os.path.join(base_path, "..")
raw_data_path = os.path.join(base_path, "raw_data")
code_path = os.path.join(base_path, "code")
log_path = os.path.join(code_path, "log")

# path to where raw nii stores
AD_raw_data_path = os.path.join(raw_data_path, "AD")
Normal_raw_data_path = os.path.join(raw_data_path, "Normal")
PD_raw_data_path = os.path.join(raw_data_path, "PD")

# path to the Makefile
make_file_path = os.path.join(code_path, "Makefile")

# path to where to save the orientation-fixed nii
preprocess_data_path = os.path.join(base_path, "preprocess_data")
AD_fixorient_path = os.path.join(preprocess_data_path, "fix_orient", "AD")
Normal_fixorient_path = os.path.join(preprocess_data_path, "fix_orient", "Normal")
PD_fixorient_path = os.path.join(preprocess_data_path, "fix_orient", "PD")


dir_check = [preprocess_data_path, AD_fixorient_path, Normal_fixorient_path, PD_fixorient_path]
for i in dir_check:
    if not os.path.exists(i):
        os.makedirs(i)

# work on PD first
print ("make sure the original nii files are back-uped and deleted!!!!!")
process_nii.fix_orientation("AD", AD_raw_data_path, AD_fixorient_path, make_file_path, log_path)
process_nii.fix_orientation("Normal", Normal_raw_data_path, Normal_fixorient_path, make_file_path, log_path)
process_nii.fix_orientation("PD", PD_raw_data_path, PD_fixorient_path, make_file_path, log_path)

