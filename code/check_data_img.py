import os
from utils import img_check

# path
base_path = os.path.abspath(os.path.dirname(__file__))
base_path = os.path.join(base_path, "..")
raw_data_path = os.path.join(base_path, "raw_data")
code_path = os.path.join(base_path, "code")
log_path = os.path.join(code_path, "log")
preprocess_data_path = os.path.join(base_path, "preprocess_data")

# path to the Makefile
make_file_path = os.path.join(code_path, "Makefile")

# path to where to save the h5
total_matrix_path = os.path.join(preprocess_data_path, "total_matrix.h5")
total_tf_matrix_path = os.path.join(preprocess_data_path, "total_tf_matrix.h5")
unique_matrix_path = os.path.join(preprocess_data_path, "unique_matrix.h5")
unique_tf_matrix_path = os.path.join(preprocess_data_path, "unique_tf_matrix.h5")

img_check.view_data(unique_tf_matrix_path, 5, "test")

