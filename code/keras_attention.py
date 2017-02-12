##### Try getting the attention map #####
##### Still going                   #####


from keras.models import load_model
import os
import numpy as np
from keras import backend as K
import cv2
import h5py
import scipy.ndimage
from matplotlib import pyplot as plt

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
#
# model_plot = os.path.join(model_path, "model_structure.png")

# rorate image
train_rotate_path = os.path.join(preprocess_data_path, "train_rotate.h5")
valid_rotate_path = os.path.join(preprocess_data_path, "valid_rotate.h5")
test_rotate_path = os.path.join(preprocess_data_path, "test_rotate.h5")

# unique test no mci
unique_test_no_mci = os.path.join(preprocess_data_path, "unique_tf_test_no_mci.h5")

model = load_model(model_body)
print("model load!")
# plot(model, to_file=model_plot)



def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    print(layer_dict.keys())
    print(layer_dict.values())
    layer = layer_dict[layer_name]
    print("we want layer")
    print(layer)
    return layer


with h5py.File(unique_test_no_mci, 'r') as hf:
    X = hf.get('X')
    Y = hf.get('Y')
    target_image = np.array(X[0,:,:,:,:]).reshape((1,96,112,96,1)) # input shape!!
    print(Y[0, :])

# get the input weights to the softmax
class_weights = model.layers[-1].get_weights()[0]
final_conv_layer = get_output_layer(model, "final_conv")
get_output = K.function([model.layers[0].input, K.learning_phase()], [final_conv_layer.output, model.layers[-1].output])

print("PASS!!!!!!!!!!!")

print(type(get_output))


print(type(target_image))
print(target_image.shape)
[conv_outputs, prediction] = get_output([target_image, 0])


print("conv_outputs!!!!!!")
print(conv_outputs.shape)
print("prediction!!!!!!")
print(prediction.shape)


conv_outputs = conv_outputs[0,:,:,:,:]
print(conv_outputs.shape)

# create activation map
cam = np.zeros(conv_outputs.shape[0:3], dtype='float32')
print("conv_outputs shape")
print(conv_outputs.shape)
print("cam shape")
print(cam.shape)
print("class weights shape")
print(class_weights.shape)
print("class weights class 0 shape")
print(class_weights[:,2].shape)

# average pooling for each feature map
avg_class_weights = np.mean(class_weights[:,2].reshape(2,3,2,64), axis=(0,1,2))
print("avg_class_weights shape")
print(avg_class_weights.shape)

# expected output
target_class = 2

for i, w in enumerate(avg_class_weights):
    print(i)
    print(w)
    cam += w * conv_outputs[:, :, :, i]

print "prediction", prediction
print("cam mean")
print(np.mean(cam))
print("cam std")
print(np.std(cam))
print("cam max")
print(np.max(cam))
print("cam min")
print(np.min(cam))

# normalize the attention map
cam = (cam - np.mean(cam))/np.std(cam)

# check point!!
print("cam shape")
print(cam.shape)
print(type(cam))
print("new cam max")
print(np.max(cam))
print("new cam min")
print(np.min(cam))
print("new cam std")
print(np.std(cam))


# zoom to the original size
cam = scipy.ndimage.zoom(cam, (48, 37.3, 48))
print("new cam")
print(cam.shape)

# target image with right size(3D)
target_img_pure = target_image.reshape(96,112,96)

# cam = cv2.resize(cam, (96, 112, 96))
heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
heatmap[np.where(cam < 0.2)] = 0
img = heatmap*0.5 + target_img_pure

# only check the middle image on z axis
middle_img = img[:,:,48] * 200
plt.imshow(middle_img)
plt.savefig("my_attention.png")



