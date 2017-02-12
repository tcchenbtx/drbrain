from keras.utils.visualize_util import plot
from keras.models import load_model
import os
import numpy as np
from keras import backend as K
import cv2
import h5py
from keras.preprocessing.image import img_to_array
from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_saliency
import scipy.ndimage
from matplotlib import pyplot as plt
import tensorflow as tf
import sys
from keras.models import Sequential
from keras.layers.core import Lambda

print (K.learning_phase())

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

unique_tf_train_path = os.path.join(preprocess_data_path, "unique_tf_train.h5")
unique_tf_valid_path = os.path.join(preprocess_data_path, "unique_tf_valid.h5")
unique_tf_test_path = os.path.join(preprocess_data_path, "unique_tf_test.h5")
# path to model
# model path

model_path = os.path.join(code_path, "models")
model_body = os.path.join(model_path, "rotate_model_new.h5")
model_weights = os.path.join(model_path, "rotate_weights_new.h5")

# rorate image
train_rotate_path = os.path.join(preprocess_data_path, "train_rotate.h5")
valid_rotate_path = os.path.join(preprocess_data_path, "valid_rotate.h5")
test_rotate_path = os.path.join(preprocess_data_path, "test_rotate.h5")

# unique test no mci
unique_test_no_mci = os.path.join(preprocess_data_path, "unique_tf_test_no_mci.h5")

model = load_model(model_body)
print("model load!")
# plot(model, to_file=model_plot)

def target_category_loss(x, category_index, nb_classes):
    return tf.mul(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def grad_cam(input_model, image, category_index, layer_name): 
    model = Sequential()
    model.add(input_model)

    nb_classes = 3
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    model.add(Lambda(target_layer,
                     output_shape = target_category_loss_output_shape))

    loss = K.sum(model.layers[-1].output)
    conv_output =  [l for l in model.layers[0].layers if l.name is layer_name][0].output
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam)



model = load_model(model_body)
print("model load!")

with h5py.File(unique_test_no_mci, 'r') as hf:
    X = hf.get('X')
    Y = hf.get('Y')
    target_image = np.array(X[0,:,:,:,:]).reshape((1,96,112,96,1)) # input shape!!
    print(Y[0, :])

predicted_class = np.argmax(model.predict(target_image))

cam  = grad_cam(model, target_image, predicted_class, "final_conv")
#cv2.imwrite("cam.jpg", cam)

