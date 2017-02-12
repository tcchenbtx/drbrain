
from __future__ import print_function

import numpy as np
import warnings
import os
from matplotlib import pyplot as plt
import h5py
from utils import data_generator

from keras.models import Model
from keras.layers import Dropout
from keras.layers import Flatten, Dense, Input, BatchNormalization, merge
from keras.layers import Convolution3D, MaxPooling3D, AveragePooling3D
from keras.preprocessing import image
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.optimizers import RMSprop, SGD



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

# model path
model_path = os.path.join(code_path, "models")
model_body = os.path.join(model_path, "V3_3D_model_no_mci.h5")
model_weights = os.path.join(model_path, "V3_3D_weights_no_mci.h5")

# figure path

with h5py.File(unique_tf_matrix_path, 'r') as hf:
    X = hf.get('X')
    Y = hf.get('Y')
    print(X.shape)
    print(Y.shape)
    train_X = X[:550,:,:,:,:]
    train_Y = Y[:550,:]
    valid_X = X[550:733,:,:,:,:]
    valid_Y = Y[550:733,:]
    test_X = X[733:,:,:,:,:]
    test_Y = Y[733:,:]

    with h5py.File(unique_tf_train_path, 'w') as ht:
        ht.create_dataset("X", data= train_X)
        ht.create_dataset("Y", data= train_Y)
    with h5py.File(unique_tf_valid_path, 'w') as hv:
        hv.create_dataset("X", data= valid_X)
        hv.create_dataset("Y", data= valid_Y)
    with h5py.File(unique_tf_test_path, 'w') as htest:
        htest.create_dataset("X", data=test_X)
        htest.create_dataset("Y", data=test_Y)


def conv3d_bn(x, nb_filter, dim1, dim2, dim3,
              border_mode='same', subsample=(1, 1, 1),
              name=None):
    '''Utility function to apply conv + BN.
    '''
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

##
##
if K.image_dim_ordering() == 'th':
    channel_axis = 1
else:
    channel_axis = 4



## model body

img_input = Input(shape=(96, 112, 96, 1))

x = conv3d_bn(img_input, 32, 3, 3, 3, subsample=(2, 2, 2), border_mode='valid')
#x = conv3d_bn(x, 32, 3, 3, 3, border_mode='valid')
x = conv3d_bn(x, 64, 3, 3, 3)
x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(x)

# prevent overfitting

x = Dropout(0.25)(x)

##############
# x = conv3d_bn(x, 80, 1, 1, 1, border_mode='valid')
# x = conv3d_bn(x, 192, 3, 3, 3, border_mode='valid')

# mixed 0, 1, 2: 35 x 35 x 256

# branch1x1 = conv3d_bn(x, 64, 1, 1, 1)
#
# branch5x5 = conv3d_bn(x, 48, 1, 1, 1)
# branch5x5 = conv3d_bn(branch5x5, 64, 5, 5, 5)
#
# branch3x3dbl = conv3d_bn(x, 64, 1, 1, 1)
# branch3x3dbl = conv3d_bn(branch3x3dbl, 96, 3, 3, 3)
#
# branch_pool = AveragePooling3D((2, 2, 2), strides=(1, 1, 1), border_mode='same')(x)
# branch_pool = conv3d_bn(branch_pool, 32, 1, 1, 1)
# x = merge([branch1x1, branch5x5, branch3x3dbl, branch_pool],
#           mode='concat', concat_axis=channel_axis, name='mixed')

# prevent overfitting

#x = Dropout(0.25)(x)

##########

# mixed 3: 17 x 17 x 768
# branch3x3 = conv3d_bn(x, 79, 3, 3, 3, subsample=(2, 2, 2), border_mode='valid')
#
# branch3x3dbl = conv3d_bn(x, 64, 1, 1, 1)
# branch3x3dbl = conv3d_bn(branch3x3dbl, 96, 3, 3, 3)
# branch3x3dbl = conv3d_bn(branch3x3dbl, 96, 3, 3, 3, subsample=(2, 2, 2), border_mode='valid')
#
# branch_pool = AveragePooling3D((3, 3, 3), strides=(2, 2, 2))(x)  # 3 -> 5
# x = merge([branch3x3, branch3x3dbl, branch_pool], mode='concat', concat_axis=channel_axis, name='mixed3')
#
# # prevent overfitting
#
# x = Dropout(0.25)(x)

###########

# mixed 4: 17 x 17 x 768
# branch1x1 = conv3d_bn(x, 79, 1, 1, 1)
#
# branch7x7 = conv3d_bn(x, 79, 1, 1, 1)
# branch7x7 = conv3d_bn(branch7x7, 79, 1, 1, 7) # 1, 7 to 1,3,7
# branch7x7 = conv3d_bn(branch7x7, 79, 1, 7, 1) # 7, 1 to 7,3,1
# branch7x7 = conv3d_bn(branch7x7, 79, 7, 1, 1) # 7, 1 to 7,3,1
#
#
# branch7x7dbl = conv3d_bn(x, 128, 1, 1, 1)
# branch7x7dbl = conv3d_bn(branch7x7dbl, 79, 7, 1, 1)
# branch7x7dbl = conv3d_bn(branch7x7dbl, 79, 1, 7, 1)
# branch7x7dbl = conv3d_bn(branch7x7dbl, 79, 1, 1, 7)
#
# branch7x7dbl = conv3d_bn(branch7x7dbl, 79, 1, 1, 7)
# branch7x7dbl = conv3d_bn(branch7x7dbl, 79, 1, 7, 1)
# branch7x7dbl = conv3d_bn(branch7x7dbl, 87, 7, 1, 1)
#
# branch_pool = AveragePooling3D((2, 2, 2), strides=(1, 1, 1), border_mode='same')(x)
# branch_pool = conv3d_bn(branch_pool, 192, 1, 1, 1)
# x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
#           mode='concat', concat_axis=channel_axis, name='mixed4')

# # mixed 5, 6: 17 x 17 x 768
#
# branch1x1 = conv3d_bn(x, 192, 1, 1, 1)
# #
# branch7x7 = conv3d_bn(x, 160, 1, 1, 1)
# branch7x7 = conv3d_bn(branch7x7, 160, 1, 1, 7)
# branch7x7 = conv3d_bn(branch7x7, 192, 1, 7, 1)
# branch7x7 = conv3d_bn(branch7x7, 192, 7, 1, 1)
# #
# branch7x7dbl = conv3d_bn(x, 160, 1, 1, 1)
# #
# branch7x7dbl = conv3d_bn(branch7x7dbl, 160, 7, 1, 1)
# branch7x7dbl = conv3d_bn(branch7x7dbl, 160, 1, 7, 1)
# branch7x7dbl = conv3d_bn(branch7x7dbl, 160, 1, 1, 7)
# #
# #
# branch7x7dbl = conv3d_bn(branch7x7dbl, 160, 1, 1, 7)
# branch7x7dbl = conv3d_bn(branch7x7dbl, 192, 1, 7, 1)
# branch7x7dbl = conv3d_bn(branch7x7dbl, 192, 7, 1, 1)


#
#
# branch_pool = AveragePooling3D((2, 2, 2), strides=(1, 1, 1), border_mode='same')(x)
# branch_pool = conv3d_bn(branch_pool, 192, 1, 1, 1)
# x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
#           mode='concat', concat_axis=channel_axis, name='mixed_5_6')

#
# # mixed 7: 17 x 17 x 768
# branch1x1 = conv3d_bn(x, 192, 1, 1, 1)
# #
# branch7x7 = conv3d_bn(x, 192, 1, 1, 1)
# branch7x7 = conv3d_bn(branch7x7, 192, 1, 3, 7)
# branch7x7 = conv3d_bn(branch7x7, 192, 7, 3, 1)
# #
# branch7x7dbl = conv3d_bn(x, 160, 1, 1, 1)
# branch7x7dbl = conv3d_bn(branch7x7dbl, 192, 7, 3, 1)
# branch7x7dbl = conv3d_bn(branch7x7dbl, 192, 1, 3, 7)
# branch7x7dbl = conv3d_bn(branch7x7dbl, 192, 7, 3, 1)
# branch7x7dbl = conv3d_bn(branch7x7dbl, 192, 1, 3, 7)
# #
# branch_pool = AveragePooling3D((3, 3, 3), strides=(1, 1, 1), border_mode='same')(x)
# branch_pool = conv3d_bn(branch_pool, 192, 1, 1, 1)
# x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
#           mode='concat', concat_axis=channel_axis, name='mixed7')
# #
# # mixed 8: 8 x 8 x 1280
# branch3x3 = conv3d_bn(x, 192, 1, 1, 1)
# branch3x3 = conv3d_bn(branch3x3, 320, 3, 3, 3, subsample=(2, 2, 2), border_mode='valid')
#
# branch7x7x3 = conv3d_bn(x, 192, 1, 1, 1)
# branch7x7x3 = conv3d_bn(branch7x7x3, 192, 1, 3, 7)
# branch7x7x3 = conv3d_bn(branch7x7x3, 192, 7, 3, 1)
# branch7x7x3 = conv3d_bn(branch7x7x3, 192, 3, 3, 3, subsample=(2, 2, 2), border_mode='valid')
#
# branch_pool = AveragePooling3D((3, 3, 3), strides=(2, 2, 2))(x)
# x = merge([branch3x3, branch7x7x3, branch_pool], mode='concat', concat_axis=channel_axis, name='mixed8')

# mixed 9: 8 x 8 x 2048
#branch1x1 = conv3d_bn(x, 79, 1, 1, 1)

#branch3x3 = conv3d_bn(x, 79, 1, 1, 1)
#branch3x3_1 = conv3d_bn(branch3x3, 79, 1, 1, 3) # 1,3 to 1, 1, 3
#branch3x3_2 = conv3d_bn(branch3x3, 79, 1, 3, 1) # 3, 1 to 3, 1, 1
#branch3x3_3 = conv3d_bn(branch3x3, 79, 3, 1, 1) # 1,3 to 1, 1, 3

#branch3x3 = merge([branch3x3_1, branch3x3_2, branch3x3_3],
#                  mode='concat', concat_axis=channel_axis, name='mixed9_1')

#branch3x3dbl = conv3d_bn(x, 120, 1, 1, 1)
#branch3x3dbl = conv3d_bn(branch3x3dbl, 79, 3, 3, 3)
#branch3x3dbl_1 = conv3d_bn(branch3x3dbl, 79, 1, 1, 3)
#branch3x3dbl_2 = conv3d_bn(branch3x3dbl, 79, 1, 3, 1)
#branch3x3dbl_3 = conv3d_bn(branch3x3dbl, 79, 3, 1, 1)

#branch3x3dbl = merge([branch3x3dbl_1, branch3x3dbl_2, branch3x3dbl_3], mode='concat', concat_axis=channel_axis)

#branch_pool = AveragePooling3D((2, 2, 2), strides=(1, 1, 1), border_mode='same')(x)
#branch_pool = conv3d_bn(branch_pool, 192, 1, 1, 1)
#x = merge([branch1x1, branch3x3, branch3x3dbl, branch_pool], mode='concat', concat_axis=channel_axis,
#          name='mixed9_2')


# x = AveragePooling3D((8, 8, 8), strides=(8, 8, 8), name='avg_pool')(x)


# reduce the feature once again
x = AveragePooling3D((3, 3, 3))(x)

########

x = Flatten(name='flatten')(x)

# prevent overfitting
# x = Dropout(0.25)(x)
#########

x = Dense(1000, activation='relu', name='my_add_dense_1')(x)

#######
#x = Dropout(0.25)(x)
######
# x = Dense(500, activation='relu', name='my_add_dense_2')(x)
x = Dense(3, activation='softmax', name='predictions')(x)



# Create model
model = Model(img_input, x)
model.summary()

sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True) # lr=0.01
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', 'precision', 'recall', 'fmeasure'])

# fit model
#history = model.fit_generator(data_generator.generator_tf(unique_tf_train_path),
#                    samples_per_epoch=500, nb_epoch=10, verbose=1,
#                    validation_data=data_generator.generator_tf(unique_tf_valid_path),
#                    nb_val_samples=32)


history = model.fit_generator(data_generator.generator_tf_no_mci(unique_tf_train_path),
                    samples_per_epoch=500, nb_epoch=20, verbose=1)




# #fit model
#history = model.fit(train_X, train_Y,
#                    batch_size=500, nb_epoch=10, verbose=1,
#                    validation_data=(valid_X, valid_Y))

# fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)


# # after fitting evaluation
#
score = model.evaluate_generator(data_generator.generator_tf_no_mci(unique_tf_valid_path),
                                val_samples=183, max_q_size=20, nb_worker=1, pickle_safe=False)


# score = model.evaluate(valid_X, valid_Y, batch_size=500, verbose=1)

# evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None)
# print score
# score = model.evaluate(valid_X_input, valid_Y_input, batch_size=3, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1]*100)


# save model weight
model.save(model_body)
model.save_weights(model_weights)


# score_tr = model.evaluate_generator(inceptv3_data_generator_3d_tf(train_x_input_file, train_y_input_file, "train_X_input", "train_Y_input"),
#                                 val_samples=32, max_q_size=20, nb_worker=1, pickle_safe=False)
#
# print('Train score:', score_tr[0])
# print('Train accuracy:', score_tr[1]*100)
#

# plot history result
print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
#plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
#plt.show()



# save model weight
# model.save(model_body)
# model.save_weights(model_weights)

# model.compile(optimizer='sgd', loss='categorical_crossentropy')
# model.fit(testx, testy, batch_size=1, nb_epoch=3, verbose=1)

