import os
from utils import process_nii

# path
base_path = os.path.abspath(os.path.dirname(__file__))
base_path = os.path.join(base_path, "..")
raw_data_path = os.path.join(base_path, "raw_data")
code_path = os.path.join(base_path, "code")
log_path = os.path.join(code_path, "log")

AD_raw_data_path = os.path.join(raw_data_path, "AD")

result = process_nii.get_nii(AD_raw_data_path)
print (result)