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
MCI_afterflirt_path = os.path.join(preprocess_data_path, "after_flirt", "MCI")
Normal_afterflirt_path = os.path.join(preprocess_data_path, "after_flirt", "Normal")
PD_afterflirt_path = os.path.join(preprocess_data_path, "after_flirt", "PD")


# get nii paths for all data

AD_nii_files = process_nii.get_nii(AD_afterflirt_path)
MCI_nii_files = process_nii.get_nii(MCI_afterflirt_path)
Normal_nii_files = process_nii.get_nii(Normal_afterflirt_path)
PD_nii_files = process_nii.get_nii(PD_afterflirt_path)


# all_data_dict = {"AD": AD_nii_files, "MCI": MCI_nii_files, "Normal": Normal_nii_files, "PD": PD_nii_files}

#(X, Y, h5file) = go_array.nii_to_1d(all_data_dict, preprocess_data_path, "total_matrix", normalization=True, smooth=1, tf=False)

#print(X.shape[0])
#print(Y.shape[0])
#print(Y[:5, :])

#del X
#del Y

#(X_tf, Y_tf, h5file_tf) = go_array.nii_to_1d(all_data_dict, preprocess_data_path, "total_tf_matrix", normalization=True, smooth=1, tf=True)

#print(X_tf.shape[0])
#print(Y_tf.shape[0])
#print(Y_tf[:5, :])

#del X_tf
#del Y_tf

# get nii paths for subject unique
#
# AD_nii_files = process_nii.get_nii(AD_afterflirt_path)
# (unique_AD_subj, unique_AD_nii_files) = go_array.unique_sub_files(AD_nii_files)
# MCI_nii_files = process_nii.get_nii(MCI_afterflirt_path)
# (unique_MCI_subj, unique_MCI_nii_files) = go_array.unique_sub_files(MCI_nii_files)
# Normal_nii_files = process_nii.get_nii(Normal_afterflirt_path)
# (unique_Normal_subj, unique_Normal_nii_files) = go_array.unique_sub_files(Normal_nii_files)
# PD_nii_files = process_nii.get_nii(PD_afterflirt_path)
# (unique_PD_subj, unique_PD_nii_files) = go_array.unique_sub_files(PD_nii_files)
#
# unique_data_dict = {"AD": unique_AD_nii_files, "MCI": unique_MCI_nii_files, "Normal": unique_Normal_nii_files, "PD": unique_PD_nii_files}
#
# #
#(unique_X, unique_Y, unique_h5file) = go_array.nii_to_1d(unique_data_dict, preprocess_data_path, "unique_matrix", normalization=True, smooth=1, tf=False)

#print(unique_X.shape[0])
#print(unique_Y.shape[0])
#print(unique_Y[:5, :])

#del unique_X
#del unique_Y
#
# (unique_X_tf, unique_Y_tf, unique_h5file_tf) = go_array.nii_to_1d(unique_data_dict, preprocess_data_path, "unique_tf_matrix", normalization=True, smooth=1, tf=True)
#
# print(unique_X_tf.shape[0])
# print(unique_Y_tf.shape[0])
# print(unique_Y_tf[:5, :])
#
# del unique_X_tf
# del unique_Y_tf
#
#

