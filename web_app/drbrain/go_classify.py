from keras.models import load_model
from drbrain import app
import os
import numpy as np
import scipy.ndimage as snd

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

model_body = os.path.join(app.config['MODLE_FOLDER'], "rotate_model_8082.h5")
model = load_model(model_body)

def normalization_data (array_1d):
    out_mean = np.mean(array_1d)
    out_std = np.std(array_1d)
    output = (array_1d - out_mean)/out_std
    print("mean:")
    print(out_mean)

    return output


def go_predict(input_img):
    print(input_img.shape)
    print("Prediction analysis")
    npad = ((3, 2), (3, 0), (3, 2))
    nii_data_pad = np.pad(input_img, pad_width=npad, mode='constant', constant_values=0)
    print(nii_data_pad.shape)
    nii_flat = np.ravel(nii_data_pad)
    nii_flat_normal = normalization_data(nii_flat)
    nii_flat_normal_smooth = snd.gaussian_filter(nii_flat_normal, 1)
    nii_flat_normal_smooth = nii_flat_normal_smooth.astype('float32')
    prep_img = nii_flat_normal_smooth.reshape((1, 96, 112, 96, 1))
    print(prep_img.shape)
    print("run prediction")
    predict_y = model.predict(prep_img)
    print("prediction done")
    return predict_y


