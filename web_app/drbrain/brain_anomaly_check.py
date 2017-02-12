
import numpy as np
from sklearn.externals import joblib
import nibabel as nib
from drbrain import app
import scipy.ndimage as snd
import os

# path info
# absolute path
app.config['APP_ROOT'] = os.path.dirname(os.path.abspath(__file__))
# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = os.path.join(app.config['APP_ROOT'], "uploads")
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['nii', 'nii.gz'])
app.config['STATIC_FOLDER'] = os.path.join(app.config['APP_ROOT'], "static")
app.config['OUTPUT_FOLDER'] = os.path.join(app.config['STATIC_FOLDER'], "output")
app.config['EXAMPLE_FOLDER'] = os.path.join(app.config['STATIC_FOLDER'], "example")
app.config['MODLE_FOLDER'] = os.path.join(app.config['STATIC_FOLDER'], "models")

#brain anomaly model load
brain_anomaly_pickle = os.path.join(app.config['MODLE_FOLDER'], "brain_anomaly.pkl")
brain_anomaly_1 = joblib.load(brain_anomaly_pickle)

# function for image normalization
def normalization_data (array_1d):
    out_mean = np.mean(array_1d)
    out_std = np.std(array_1d)
    output = (array_1d - out_mean)/out_std
    print("mean:")
    print(out_mean)

    return output

# function to run brain anomaly check
def run_brain_anomaly(beauty_nii_array):
    # open file
    print(beauty_nii_array.shape)
    npad = ((3, 2), (3, 0), (3, 2))
    nii_data_pad = np.pad(beauty_nii_array, pad_width=npad, mode='constant', constant_values=0)
    nii_flat = np.ravel(nii_data_pad)
    nii_flat_normal = normalization_data(nii_flat)
    nii_flat_normal_smooth = snd.gaussian_filter(nii_flat_normal, 1)
    nii_flat_normal_smooth = nii_flat_normal_smooth.astype('float32')
    print(nii_flat_normal_smooth.shape)

    anomaly_outcome = brain_anomaly_1.predict(nii_flat_normal_smooth)
    return anomaly_outcome


