import h5py
import numpy as np

# to deal with large dataset, need data generator for keras

# randomize data and output in tensorflow shape (for 4 categories)
def generator_tf(h5_file):
    while 1:
        with h5py.File(h5_file, 'r') as hf:
            X = hf.get("X")
            Y = hf.get("Y")
            shuffle = np.random.permutation(X.shape[0]).argsort()
            for i in shuffle:
                x_array = X[i, :, :, :, :].reshape(1, 96, 112, 96, 1)
                x_array = x_array.astype('float32')
                y_array = Y[i, :].reshape(1,4)
                #print(y_array)
                y_array = y_array.astype('float32')
                yield (x_array, y_array)

# don't randomize, output in tensorflow shape (for 4 categories)
def generator_tf_valid(h5_file):
    while 1:
        with h5py.File(h5_file, 'r') as hf:
            X = hf.get("X")
            Y = hf.get("Y")
            for i in range(X.shape[0]):
                x_array = X[i, :, :, :, :].reshape(1, 96, 112, 96, 1)
                x_array = x_array.astype('float32')
                y_array = Y[i, :].reshape(1,4)
                print(y_array)
                y_array = y_array.astype('float32')
                yield (x_array, y_array)

# randomize data and output in tensorflow shape (for 3 categories)
def generator_tf_no_mci(h5_file):
    while 1:
        with h5py.File(h5_file, 'r') as hf:
            X = hf.get("X")
            Y = hf.get("Y")
            shuffle = np.random.permutation(X.shape[0]).argsort()
            for i in shuffle:
                x_array = X[i, :, :, :, :].reshape(1, 96, 112, 96, 1)
                x_array = x_array.astype('float32')
                y_array = Y[i, :].reshape(1,3)
                y_array = y_array.astype('float32')
                yield (x_array, y_array)

# don't randomize, output in tensorflow shape (for 3 categories)
def generator_tf_no_mci_order(h5_file):
    while 1:
        with h5py.File(h5_file, 'r') as hf:
            X = hf.get("X")
            Y = hf.get("Y")
            for i in range(X.shape[0]):
                x_array = X[i, :, :, :, :].reshape(1, 96, 112, 96, 1)
                x_array = x_array.astype('float32')
                y_array = Y[i, :].reshape(1,3)
                y_array = y_array.astype('float32')
                yield (x_array, y_array)
