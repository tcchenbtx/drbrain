from __future__ import print_function

import os
from matplotlib import pyplot as plt
from utils import data_generator
from keras.models import Model
from keras.layers import Dropout
from keras.layers import Flatten, Dense, Input, BatchNormalization, merge
from keras.layers import Convolution3D, MaxPooling3D, AveragePooling3D
from keras import backend as K
from keras.optimizers import RMSprop, SGD
import json


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
total_matrix_path = os.path.join(preprocess_data_path, "total_matrix_no_mci.h5")
total_tf_matrix_path = os.path.join(preprocess_data_path, "total_tf_matrix_no_mci.h5")
unique_matrix_path = os.path.join(preprocess_data_path, "unique_matrix_no_mci.h5")
unique_tf_matrix_path = os.path.join(preprocess_data_path, "unique_tf_matrix_no_mci.h5")

unique_tf_train_path = os.path.join(preprocess_data_path, "unique_tf_train_no_mci.h5")
unique_tf_valid_path = os.path.join(preprocess_data_path, "unique_tf_valid_no_mci.h5")
unique_tf_test_path = os.path.join(preprocess_data_path, "unique_tf_test_no_mci.h5")

# model path (where to save the model)
model_path = os.path.join(code_path, "models")
model_body = os.path.join(model_path, "rotate_model.h5")
model_weights = os.path.join(model_path, "rotate_weights.h5")

# dataset for train/validate/test
train_rotate_path = os.path.join(preprocess_data_path, "train_rotate.h5")
valid_rotate_path = os.path.join(preprocess_data_path, "valid_rotate.h5")
test_rotate_path = os.path.join(preprocess_data_path, "test_rotate.h5")


# function to apply 3D convolution + batch-normalization (adopted from keras models)
def conv3d_bn(x, nb_filter, dim1, dim2, dim3,
              border_mode='same', subsample=(1, 1, 1),
              name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_dim_ordering() == 'th':
        bn_axis = 1
    else:
        bn_axis = 4
    x = Convolution3D(nb_filter, dim1, dim2, dim3,
                      subsample=subsample,
                      activation='relu',
                      border_mode=border_mode,
                      name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name)(x)
    return x


# make sure the channel axis info is correct
if K.image_dim_ordering() == 'th':
    channel_axis = 1
else:
    channel_axis = 4



## model body

img_input = Input(shape=(96, 112, 96, 1))

x = conv3d_bn(img_input, 32, 3, 3, 3, subsample=(2, 2, 2), border_mode='valid')
x = conv3d_bn(x, 32, 3, 3, 3, border_mode='valid')
x = conv3d_bn(x, 64, 3, 3, 3)
x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(x)

# prevent overfitting
x = Dropout(0.25)(x)

##############
x = conv3d_bn(x, 80, 1, 1, 1, border_mode='valid')
x = conv3d_bn(x, 192, 3, 3, 3, border_mode='valid')

# add inception module
branch1x1 = conv3d_bn(x, 64, 1, 1, 1)
#
branch5x5 = conv3d_bn(x, 48, 1, 1, 1)
branch5x5 = conv3d_bn(branch5x5, 64, 5, 5, 5)
#
branch3x3dbl = conv3d_bn(x, 64, 1, 1, 1)
branch3x3dbl = conv3d_bn(branch3x3dbl, 96, 3, 3, 3)
#
branch_pool = AveragePooling3D((2, 2, 2), strides=(1, 1, 1), border_mode='same')(x)
branch_pool = conv3d_bn(branch_pool, 32, 1, 1, 1)
x = merge([branch1x1, branch5x5, branch3x3dbl, branch_pool],
          mode='concat', concat_axis=channel_axis, name='mixed')

x = conv3d_bn(x, 64, 3, 3, 3, name="final_conv")

# reduce the feature once again
x = AveragePooling3D((3, 3, 3))(x)


x = Flatten(name='flatten')(x)
x = Dense(1000, activation='relu', name='my_add_dense_1')(x)
x = Dense(3, activation='softmax', name='predictions')(x)



# Create model
model = Model(img_input, x)
model.summary()

sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', 'precision', 'recall', 'fmeasure'])


history = model.fit_generator(data_generator.generator_tf_no_mci(train_rotate_path),
                    samples_per_epoch=1000, nb_epoch=20, verbose=1)



# evaluation set:
score = model.evaluate_generator(data_generator.generator_tf_no_mci(valid_rotate_path),
                                val_samples=1656, max_q_size=1, nb_worker=1, pickle_safe=False)

print('Valid score:', score[0])
print('Valid accuracy:', score[1]*100)

# test set:
score2 = model.evaluate_generator(data_generator.generator_tf_no_mci(test_rotate_path),
                                val_samples=1656, max_q_size=1, nb_worker=1, pickle_safe=False)
print('Test score:', score2[0])
print('Test accuracy:', score2[1]*100)


# what's in the history?
print(history.history.keys())

# save the history dictionary
with open("history.json", "w") as uoutput:
    json.dump(history.history, uoutput, indent=4, sort_keys=True)


# plot the training accuracy:
fig1 = plt.figure(figsize=(8, 6))
plt.plot(history.history['acc'])
plt.title('Model accuracy', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.xlabel('Iteration', fontsize=20)
plt.savefig("training_accuracy.png")
plt.close

# plot the training loss:
fig2 = plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'])
plt.title('Model loss', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.xlabel('Iteration', fontsize=15)
plt.savefig("training_loss.png")
plt.close


# save model weight
model.save(model_body)
model.save_weights(model_weights)

