from utils import process_dcm
import os

# path
base_path = os.path.abspath(os.path.dirname(__file__))
base_path = os.path.join(base_path, "..")
raw_data_path = os.path.join(base_path, "raw_data")
code_path = os.path.join(base_path, "code")

# path to process dcm files
AD_raw_data_path = os.path.join(raw_data_path, "AD")
make_file_path = os.path.join(code_path, "Makefile")


process_dcm.batch_dcm2nii(AD_raw_data_path, make_file_path)
