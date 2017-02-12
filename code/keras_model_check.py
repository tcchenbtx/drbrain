from keras.utils.visualize_util import plot
from keras.models import load_model
import os
import numpy as np
import h5py
from utils import data_generator

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

# rorate image
train_rotate_path = os.path.join(preprocess_data_path, "train_rotate.h5")
valid_rotate_path = os.path.join(preprocess_data_path, "valid_rotate.h5")
test_rotate_path = os.path.join(preprocess_data_path, "test_rotate.h5")

# check size
with h5py.File(valid_rotate_path, 'r') as hf:
    X = hf.get('X')
    Y = hf.get('Y')
    print(X.shape)
    print(Y.shape)

with h5py.File(test_rotate_path, 'r') as ht:
    X = ht.get('X')
    Y = ht.get('Y')
    print(X.shape)
    print(Y.shape)

# model path
model_path = os.path.join(code_path, "models")
model_body = os.path.join(model_path, "rotate_model.h5")
model_weights = os.path.join(model_path, "rotate_weights.h5")

# where to save model structure
model_plot = os.path.join(model_path, "model_structural.png")

# load model
model = load_model(model_body)
plot(model, to_file=model_plot)
print('Model loaded.')

# check validation accuracy
score = model.evaluate_generator(data_generator.generator_tf_no_mci(valid_rotate_path),
                                val_samples=1656, max_q_size=20, nb_worker=1, pickle_safe=False)

print('Valid score:', score[0])
print('Valid accuracy:', score[1]*100)

# check test accuracy
score = model.evaluate_generator(data_generator.generator_tf_no_mci(test_rotate_path),
                                val_samples=1656, max_q_size=20, nb_worker=1, pickle_safe=False)

print('Test score:', score[0])
print('Test accuracy:', score[1]*100)


# with rotate test 

predict_y = model.predict_generator(data_generator.generator_tf_no_mci_order(test_rotate_path),
                                    val_samples=1656, max_q_size=10, nb_worker=1, pickle_safe=False)


with h5py.File(test_rotate_path, 'r') as hf:
    X = hf.get('X')
    Y = hf.get('Y')
    real_y = np.array(Y[:,:])
    print("unique tf test")
    print(X.shape)
    print(Y.shape)


print(real_y.shape)
print(predict_y.shape)


real_y_list = []
predict_y_list = []

for i in range(real_y.shape[0]):
    real_y_list.append(np.argmax(real_y[i,:]))
    predict_y_list.append(np.argmax(predict_y[i, :]))


print(real_y_list)
print(predict_y_list)


# for precision recall metrics
from sklearn import metrics
myprecision = metrics.precision_score(real_y_list, predict_y_list, average='macro')
myrecall = metrics.recall_score(real_y_list, predict_y_list, average='micro')
print(myprecision)
print(myrecall)

from sklearn.metrics import precision_recall_fscore_support as score
precision, recall, fscore, support = score(real_y_list, predict_y_list)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

