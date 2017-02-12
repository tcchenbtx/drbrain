
from __future__ import print_function

from __future__ import division, print_function
import os
import numpy as np
import h5py
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


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

with h5py.File(unique_matrix_path, 'r') as hf:
    X = hf.get('X')
    Y = hf.get('Y')
    Y_num = hf.get('Y_num')
    print(X.shape)
    print(Y.shape)
    print(len(Y_num))

    train_subset_X = X[:752, :]
    train_subset_Y = Y[:752]
    valid_subset_X = X[752:,:]
    valid_subset_Y = Y[752:]


# build model
model = Sequential()
model.add(Dropout(0.5, input_shape=(1032192,)))
model.add(Dense(10000, init='uniform', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(100, init='uniform', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(4, init='uniform', activation='softmax'))


model.summary()

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model
model.fit(train_subset_X, train_subset_Y, nb_epoch=10, batch_size=1)

# evaluate the model
scores = model.evaluate(valid_subset_X, valid_subset_Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

