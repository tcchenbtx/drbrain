import os
from utils import process_nii

# path
base_path = os.path.abspath(os.path.dirname(__file__))
base_path = os.path.join(base_path, "..")
raw_data_path = os.path.join(base_path, "raw_data")
code_path = os.path.join(base_path, "code")
log_path = os.path.join(code_path, "log")
preprocess_data_path = os.path.join(base_path, "preprocess_data")

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

# path to the Makefile
make_file_path = os.path.join(code_path, "Makefile")

# path to where to save the post-fnirt nii
AD_afterfnirt_path = os.path.join(preprocess_data_path, "after_fnirt", "AD")
MCI_afterfnirt_path = os.path.join(preprocess_data_path, "after_fnirt", "MCI")
Normal_afterfnirt_path = os.path.join(preprocess_data_path, "after_fnirt", "Normal")
PD_afterfnirt_path = os.path.join(preprocess_data_path, "after_fnirt", "PD")


dir_check = [AD_afterfnirt_path, MCI_afterfnirt_path, Normal_afterfnirt_path, PD_afterfnirt_path]
for i in dir_check:
    if not os.path.exists(i):
        os.makedirs(i)

# work on PD first
#process_nii.apply_fnirt("AD1", AD_afterflirt_path, AD_afterfnirt_path, AD_affmatrix_path, make_file_path, log_path)
#process_nii.apply_fnirt("MCI1", MCI_afterflirt_path, MCI_afterfnirt_path, MCI_affmatrix_path, make_file_path, log_path)
#process_nii.apply_fnirt("Normal1", Normal_afterflirt_path, Normal_afterfnirt_path, Normal_affmatrix_path, make_file_path, log_path)
process_nii.apply_fnirt("PD1", PD_afterflirt_path, PD_afterfnirt_path, PD_affmatrix_path, make_file_path, log_path)

