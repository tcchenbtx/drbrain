import os
from utils import process_nii

# path
base_path = os.path.abspath(os.path.dirname(__file__))
base_path = os.path.join(base_path, "..")
raw_data_path = os.path.join(base_path, "raw_data")
code_path = os.path.join(base_path, "code")
preprocess_data_path = os.path.join(base_path, "preprocess_data")

# path to process dcm files
AD_raw_data_path = os.path.join(raw_data_path, "AD")
MCI_raw_data_path = os.path.join(raw_data_path, "MCI")
Normal_raw_data_path = os.path.join(raw_data_path, "Normal")
PD_raw_data_path = os.path.join(raw_data_path, "PD")

make_file_path = os.path.join(code_path, "Makefile")

#

# path to reoriented data
AD_fixorient_path = os.path.join(preprocess_data_path, "fix_orient", "AD")
MCI_fixorient_path = os.path.join(preprocess_data_path, "fix_orient", "MCI")
Normal_fixorient_path = os.path.join(preprocess_data_path, "fix_orient", "Normal")
PD_fixorient_path = os.path.join(preprocess_data_path, "fix_orient", "PD")

# path to where to save the cropped nii
AD_crop_path = os.path.join(preprocess_data_path, "crop_figure", "AD")
MCI_crop_path = os.path.join(preprocess_data_path, "crop_figure", "MCI")
Normal_crop_path = os.path.join(preprocess_data_path, "crop_figure", "Normal")
PD_crop_path = os.path.join(preprocess_data_path, "crop_figure", "PD")

# skull stripped
AD_skullstrip_path = os.path.join(preprocess_data_path, "skull_strip", "AD")
MCI_skullstrip_path = os.path.join(preprocess_data_path, "skull_strip", "MCI")
Normal_skullstrip_path = os.path.join(preprocess_data_path, "skull_strip", "Normal")
PD_skullstrip_path = os.path.join(preprocess_data_path, "skull_strip", "PD")

# path to where to save the post-flirt nii
AD_afterflirt_path = os.path.join(preprocess_data_path, "after_flirt", "AD")
MCI_afterflirt_path = os.path.join(preprocess_data_path, "after_flirt", "MCI")
Normal_afterflirt_path = os.path.join(preprocess_data_path, "after_flirt", "Normal")
PD_afterflirt_path = os.path.join(preprocess_data_path, "after_flirt", "PD")

AD_afterfnirt_path = os.path.join(preprocess_data_path, "after_fnirt", "AD")
MCI_afterfnirt_path = os.path.join(preprocess_data_path, "after_fnirt", "MCI")
Normal_afterfnirt_path = os.path.join(preprocess_data_path, "after_fnirt", "Normal")
PD_afterfnirt_path = os.path.join(preprocess_data_path, "after_fnirt", "PD")


# visualize raw data
#
AD1 = process_nii.get_nii(AD_raw_data_path)
MCI1 = process_nii.get_nii(MCI_raw_data_path)
Normal1 = process_nii.get_nii(Normal_raw_data_path)
PD1 = process_nii.get_nii(PD_raw_data_path)
my_dict = {"AD":AD1, "MCI": MCI1, "PD":PD1, "Normal": Normal1}

process_nii.random_visualization(5, my_dict, "raw_data")


# visualize reoriented data

# AD1_reorient = process_nii.get_nii(AD_fixorient_path)
# MCI1_reorient = process_nii.get_nii(MCI_fixorient_path)
# Normal1_reorient = process_nii.get_nii(Normal_fixorient_path)
# PD1_reorient = process_nii.get_nii(PD_fixorient_path)
# my_dict_reorient = {"AD":AD1_reorient, "MCI": MCI1_reorient, "PD": PD1_reorient, "Normal": Normal1_reorient}
# process_nii.random_visualization(5, my_dict_reorient, "test_reorient")

# visualize skull stripped data

# AD1_skull = process_nii.get_nii(AD_skullstrip_path)
# MCI1_skull = process_nii.get_nii(MCI_skullstrip_path)
# Normal1_skull = process_nii.get_nii(Normal_skullstrip_path)
# my_dict_skull = {"AD1":AD1_skull, "MCI1": MCI1_skull, "Mormal1": Normal1_skull}
#
# process_nii.random_visualization(5, my_dict_skull, "test_skullstrip")
#

# PD only visualization
# PD_raw = process_nii.get_nii(PD_raw_data_path)
# PD_reorient = process_nii.get_nii(PD_fixorient_path)
# PD_crop = process_nii.get_nii(PD_crop_path)
# PD_skullstrip = process_nii.get_nii(PD_skullstrip_path)
# PD_afterflirt = process_nii.get_nii(PD_afterflirt_path)
# PD_afterfnirt = process_nii.get_nii(PD_afterfnirt_path)
# my_dict_reorient = {"raw":PD_raw, "skullstrip": PD_skullstrip, "afteraff":PD_afterflirt, "fnirt": PD_afterfnirt}
# process_nii.fix_visualization(5, my_dict_reorient, "PD_process")

# visualize stdbrain data
# AD1 = process_nii.get_nii(AD_afterflirt_path)
# MCI1 = process_nii.get_nii(MCI_afterflirt_path)
# Normal1 = process_nii.get_nii(Normal_afterflirt_path)
# PD1 = process_nii.get_nii(PD_afterflirt_path)
# my_dict = {"AD":AD1, "MCI": MCI1, "PD":PD1, "Normal": Normal1}
# #
# process_nii.random_visualization(5, my_dict, "afterflirt_data")