
from keras.models import load_model
#from drbrain import app
import os
import nibabel as nib
import numpy as np

# path info
# absolute path
app = os.path.dirname(os.path.abspath(__file__))
# This is the path to the upload directory
#appconfig['UPLOAD_FOLDER'] = os.path.join(app.config['APP_ROOT'], "uploads")
# These are the extension that we are accepting to be uploaded
#app.config['ALLOWED_EXTENSIONS'] = set(['nii', 'nii.gz'])
#app.config['STATIC_FOLDER'] = os.path.join(app.config['APP_ROOT'], "static")
#app.config['OUTPUT_FOLDER'] = os.path.join(app.config['STATIC_FOLDER'], "output")
#app.config['EXAMPLE_FOLDER'] = os.path.join(app.config['STATIC_FOLDER'], "example")
#app.config['MODLE_FOLDER'] = os.path.join(app.config['STATIC_FOLDER'], "models")



models_path = os.path.join(app, "static", "models")
upload_path = os.path.join(app, "uploads")



model_body = os.path.join(models_path, "rotate_model_8082.h5")
model = load_model(model_body)

def go_predict(input_img):
    predict_y = model.predict(input_img)
    return predict_y


nii_load = nib.load(os.path.join(upload_path, "ADNI_116_S_4043_S117902_stdbrain.nii.gz"))
nii_data = nii_load.get_data()

print("nii original")
print(nii_data.shape)

npad = ((3, 2), (3, 0), (3, 2))
nii_data_pad = np.pad(nii_data, pad_width=npad, mode='constant', constant_values=0)


print("after pad")
print(nii_data_pad.shape)

prep = nii_data_pad.reshape((1, 96, 112, 96, 1))
out = go_predict(prep)
print(out)


