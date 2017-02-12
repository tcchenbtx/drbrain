from utils import process_dcm
import os

# path
base_path = os.path.abspath(os.path.dirname(__file__))
base_path = os.path.join(base_path, "..")
raw_data_path = os.path.join(base_path, "raw_data")
code_path = os.path.join(base_path, "code")
log_path = os.path.join(code_path, "log")

# path to process dcm files
AD_raw_data_path = os.path.join(raw_data_path, "AD")
Normal_raw_data_path = os.path.join(raw_data_path, "Normal")
PD_raw_data_path = os.path.join(raw_data_path, "PD")

# path to makefile
make_file_path = os.path.join(code_path, "Makefile")

# go!
process_dcm.batch_dcm2nii("AD1", AD_raw_data_path, make_file_path, log_path)
process_dcm.batch_dcm2nii("Normal1", Normal_raw_data_path, make_file_path, log_path)
process_dcm.batch_dcm2nii("PD1", PD_raw_data_path, make_file_path, log_path)

