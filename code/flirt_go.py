import os
from utils import process_nii

# path
base_path = os.path.abspath(os.path.dirname(__file__))
base_path = os.path.join(base_path, "..")
raw_data_path = os.path.join(base_path, "raw_data")
code_path = os.path.join(base_path, "code")
log_path = os.path.join(code_path, "log")

# path to where to save the skull strip nii
preprocess_data_path = os.path.join(base_path, "preprocess_data")
AD_skullstrip_path = os.path.join(preprocess_data_path, "skull_strip", "AD")
MCI_skullstrip_path = os.path.join(preprocess_data_path, "skull_strip", "MCI")
Normal_skullstrip_path = os.path.join(preprocess_data_path, "skull_strip", "Normal")
PD_skullstrip_path = os.path.join(preprocess_data_path, "skull_strip", "PD")

# path to the Makefile
make_file_path = os.path.join(code_path, "Makefile")

# path to where to save the post-flirt nii
AD_afterflirt_path = os.path.join(preprocess_data_path, "after_flirt", "AD")
MCI_afterflirt_path = os.path.join(preprocess_data_path, "after_flirt", "MCI")
Normal_afterflirt_path = os.path.join(preprocess_data_path, "after_flirt", "Normal")
PD_afterflirt_path = os.path.join(preprocess_data_path, "after_flirt", "PD")

# path to where to save the matrix
AD_affmatrix_path = os.path.join(preprocess_data_path, "affine_matrix", "AD")
MCI_affmatrix_path = os.path.join(preprocess_data_path, "affine_matrix", "MCI")
Normal_affmatrix_path = os.path.join(preprocess_data_path, "affine_matrix", "Normal")
PD_affmatrix_path = os.path.join(preprocess_data_path, "affine_matrix", "PD")



dir_check = [AD_afterflirt_path, MCI_afterflirt_path, Normal_afterflirt_path, PD_afterflirt_path,
             AD_affmatrix_path, MCI_affmatrix_path, Normal_affmatrix_path, PD_affmatrix_path]
for i in dir_check:
    if not os.path.exists(i):
        os.makedirs(i)

# work on PD first
#process_nii.apply_flirt("AD1", AD_skullstrip_path, AD_afterflirt_path, AD_affmatrix_path, make_file_path, log_path)
#process_nii.apply_flirt("MCI1", MCI_skullstrip_path, MCI_afterflirt_path, MCI_affmatrix_path, make_file_path, log_path)
process_nii.apply_flirt("Normal", Normal_skullstrip_path, Normal_afterflirt_path, Normal_affmatrix_path, make_file_path, log_path)
#process_nii.apply_flirt("PD1", PD_skullstrip_path, PD_afterflirt_path, PD_affmatrix_path, make_file_path, log_path)

