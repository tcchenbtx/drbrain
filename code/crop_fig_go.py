import os
from utils import process_nii

# path
base_path = os.path.abspath(os.path.dirname(__file__))
base_path = os.path.join(base_path, "..")
raw_data_path = os.path.join(base_path, "raw_data")
code_path = os.path.join(base_path, "code")
log_path = os.path.join(code_path, "log")

# path to where re-oriented nii stores
preprocess_data_path = os.path.join(base_path, "preprocess_data")
AD_fixorient_path = os.path.join(preprocess_data_path, "fix_orient", "AD")
MCI_fixorient_path = os.path.join(preprocess_data_path, "fix_orient", "MCI")
Normal_fixorient_path = os.path.join(preprocess_data_path, "fix_orient", "Normal")
PD_fixorient_path = os.path.join(preprocess_data_path, "fix_orient", "PD")

# path to the Makefile
make_file_path = os.path.join(code_path, "Makefile")

# path to where to save the cropped nii
AD_crop_path = os.path.join(preprocess_data_path, "crop_figure", "AD")
MCI_crop_path = os.path.join(preprocess_data_path, "crop_figure", "MCI")
Normal_crop_path = os.path.join(preprocess_data_path, "crop_figure", "Normal")
PD_crop_path = os.path.join(preprocess_data_path, "crop_figure", "PD")


dir_check = [AD_crop_path, MCI_crop_path, Normal_crop_path, PD_crop_path]
for i in dir_check:
    if not os.path.exists(i):
        os.makedirs(i)

# work on PD first
# process_nii.crop_fig("AD", AD_fixorient_path, AD_crop_path, make_file_path, log_path)
# process_nii.crop_fig("MCI1", MCI_fixorient_path, MCI_crop_path, make_file_path, log_path)
process_nii.crop_fig("Normal", Normal_fixorient_path, Normal_crop_path, make_file_path, log_path)
# process_nii.crop_fig("PD1", PD_fixorient_path, PD_crop_path, make_file_path, log_path)




