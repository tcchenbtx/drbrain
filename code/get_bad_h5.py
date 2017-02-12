import os
from utils import process_nii
from utils import go_array

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


# Bad files
Bad_afterflirt_path = os.path.join(preprocess_data_path, "after_flirt", "Bad")
Bad_to_h5 = os.path.join(preprocess_data_path, "Real_bad_matrix.h5")

# get nii paths for bad data
# convert to subjects x 1d array

Bad_nii_files = process_nii.get_nii(Bad_afterflirt_path)
bad_matrix = go_array.nii_to_1d_simple(Bad_nii_files, Bad_to_h5, normalization=True, smooth=1, tf=False)

print(bad_matrix.shape)
print(bad_matrix.dtype)