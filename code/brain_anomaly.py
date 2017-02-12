
from utils import process_nii
from utils import go_array
import h5py
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from sklearn.externals import joblib

# path
base_path = os.path.abspath(os.path.dirname(__file__))
base_path = os.path.join(base_path, "..")
raw_data_path = os.path.join(base_path, "raw_data")
code_path = os.path.join(base_path, "code")
log_path = os.path.join(code_path, "log")
preprocess_data_path = os.path.join(base_path, "preprocess_data")
models_path = os.path.join(code_path, "models")

# path to the Makefile
make_file_path = os.path.join(code_path, "Makefile")

# path to where to save the h5
total_matrix_path = os.path.join(preprocess_data_path, "total_matrix.h5")
total_tf_matrix_path = os.path.join(preprocess_data_path, "total_tf_matrix.h5")
unique_matrix_path = os.path.join(preprocess_data_path, "unique_matrix.h5")
unique_tf_matrix_path = os.path.join(preprocess_data_path, "unique_tf_matrix.h5")

bad_matrix_path = os.path.join(preprocess_data_path, "bad_matrix.h5")

# where to save model
brain_anomaly_model_path = os.path.join(models_path, "brain_anomaly.pkl")

# create bad data matrix (only need to run once!!)
Bad_afterflirt_path = os.path.join(preprocess_data_path, "after_flirt", "Bad")
Bad_nii_files = process_nii.get_nii(Bad_afterflirt_path)
bad_data_dict = {"Bad": Bad_nii_files}

bad_data_matrix = go_array.nii_to_1d_simple(bad_data_dict, bad_matrix_path, normalization=True, smooth=1, tf=False)

print("bad matrix:")
print(bad_data_matrix.dtype)
print(bad_data_matrix.shape)

# get all good data

with h5py.File(unique_matrix_path, 'r') as hf:
    good = hf.get('X')
    good_X = np.array(good)

with h5py.File(bad_matrix_path, 'r') as hb:
    bad = hb.get('X')
    bad_X = bad[0,:].reshape((1,96*112*96))
    print("bad shape")
    print(bad.shape)
    for i in range(1, 52):
        if np.isnan(np.sum(bad[i, :])):
            print("have nan!!")
        else:
            go_bad = bad[i,:].reshape(1, 96*112*96)
            bad_X = np.vstack((bad_X, go_bad))
            print("OK")

    print("bad_X shape")
    print(bad_X.shape)
    


print("bad data:")
print(bad_X.dtype)
print(bad_X.shape)
print("good data:")
print(good_X.dtype)
print(good_X.shape)



y_label_list = ([0] * good_X.shape[0]) + ([1] * bad_X.shape[0])
y_label = np.array(y_label_list)

real_y_for_anomaly = ([1] * good_X.shape[0]) + ([-1] * bad_X.shape[0])
real_y_for_anomaly = np.array(real_y_for_anomaly)

X = np.vstack((good_X, bad_X))
print("X shape")
print(X.shape)
print("y_lable shape")
print(y_label.shape)


#### Anomaly Detection

# train with part of the good_X
X_train = good_X[:1025,:]
print("X_train:")
print(X_train.shape)

# test with rest of the good_X and all bad_X
X_test = np.vstack((good_X[1025:, :], bad_X))
print("X_test:")
print(X_test.shape)

# expected outcome
real_outcome = ([1] * good_X[1025:,:].shape[0]) + ([0] * bad_X.shape[0])
real_outcome = np.array(real_outcome)
print(real_outcome.shape)

# double check the train and test set
print("X_train")
print(X_train.shape)
print("X_test")
print(X_test.shape)

# Isolation Forest model
rng = np.random.RandomState(42)
brain_anomaly = IsolationForest(max_samples='auto', random_state=rng)
brain_anomaly.fit(X_train)

# run on X_test:
# 1 is good
# -1 is bad
predict = brain_anomaly.predict(X_test)

# get predicted_bad and predicted_good
predict_array = np.array(predict)
predict_bad = X_test[predict_array == -1]
predict_good = X_test[predict_array == 1]

# plot predicted bad and predicted good
fig = plt.figure()
fig_indx = 0
for i in range(predict_bad.shape[0]):
    nii_data = predict_bad[i]
    nii_data = nii_data.reshape((96, 112, 96))
    plt.subplot(6, 10, fig_indx, xticks=[], yticks=[])
    plt.imshow(nii_data[:,:, 48], cmap='gray')
    fig_indx += 1
plt.savefig("model_found_bad_again.png")
plt.close()


fig = plt.figure()
fig_indx = 0
for i in range(predict_good.shape[0]):
    nii_data = predict_good[i]
    nii_data = nii_data.reshape((96, 112, 96))
    plt.subplot(6, 10, fig_indx, xticks=[], yticks=[])
    plt.imshow(nii_data[:,:, 48], cmap='gray')
    fig_indx += 1
plt.savefig("model_found_good_again.png")
plt.close()

# output the predicted result
print (predict)

# save model

joblib.dump(brain_anomaly, brain_anomaly_model_path)




