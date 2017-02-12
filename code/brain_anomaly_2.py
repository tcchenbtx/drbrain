from sklearn.mixture import GaussianMixture
from utils import process_nii
from utils import go_array
import h5py
import os
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.datasets import load_boston
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
# from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

brain_anomaly_model_path = os.path.join(models_path, "brain_anomaly_2d.pkl")

# get all good data

with h5py.File(unique_matrix_path, 'r') as hf:
    good = hf.get('X')
    good_X = np.zeros((good.shape[0], 112*96))
    for i in range(good.shape[0]):
        get_nii = good[i, :].reshape((96, 112, 96))
        good_X[i, :] = np.ravel(get_nii[:, :, 48])
    print("good_X shape:")
    print(good_X.shape)


with h5py.File(bad_matrix_path, 'r') as hb:
    bad = hb.get('X')
    get_nii = bad[0, :].reshape((96, 112, 96))
    bad_X = np.ravel(get_nii[:,:,48])
    print("bad shape")
    print(bad.shape)
    for i in range(1, 52):
        if np.isnan(np.sum(bad[i, :])):
            print("have nan!!")
        else:
            go_bad = bad[i, :].reshape((96, 112, 96))
            go_bad = np.ravel(go_bad[:, :, 48])
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



y_label = ([0] * good_X.shape[0]) + ([1] * bad_X.shape[0])
y_label = np.array(y_label)

real_y_for_anomaly = ([1] * good_X.shape[0]) + ([-1] * bad_X.shape[0])
real_y_for_anomaly = np.array(real_y_for_anomaly)

X = np.vstack((good_X, bad_X))
print("X shape")
print(X.shape)
print("y_lable shape")
print(y_label.shape)

# ### PCA plot
# target_names = ['Good', 'Bad']
#
# pca = PCA(n_components=2)
# X_r = pca.fit(good_X).transform(X)
#
#
# # Percentage of variance explained for each components
# print('explained variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))
#
# plt.figure()
# colors = ['navy', 'magenta']
# lw = 1
#
#
# ## Good = 0, Bad = 1
#
# for color, i, target_name in zip(colors, [0, 1], target_names):
#     plt.scatter(X_r[y_label == i, 0], X_r[y_label == i, 1], color=color, alpha=.8, lw=lw, label=target_name, s=10)
#     plt.legend(loc='best', shadow=False, scatterpoints=1)
#
# plt.title('Abnormal Data')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.savefig("Abnormal_data_PCA_on_2d.png")



#### abnormaly
# # train with X_good
#
X_train = good_X[:1025, :]
print("X_train:")
print(X_train.shape)
X_test = np.vstack((good_X[1025:, :], bad_X))
print("X_test:")
print(X_test.shape)

real_outcome = ([1] * good_X[1025:,:].shape[0]) + ([0] * bad_X.shape[0])
real_outcome = np.array(real_outcome)
print(real_outcome.shape)
#
# print("X_train")
# print(X_train.shape)
# print("X_test")
# print(X_test.shape)
#
#
# clf = IsolationForest(max_samples='auto', random_state=rng)
# clf.fit(X_train)
# y_pred_train = clf.predict(X_train)
# y_pred_test = clf.predict(X_test)

rng = np.random.RandomState(60) # 42, 60
clf = IsolationForest(max_samples='auto', random_state=rng)
clf.fit(X_train)


# 1 is good
# -1 is bad
predict = clf.predict(X_test)

predict_array = np.array(predict)
predict_bad = X_test[predict_array == -1]
predict_good = X_test[predict_array == 1]

# fig = plt.figure()
# fig_indx = 0
# for i in range(predict_bad.shape[0]):
#     nii_data = predict_bad[i]
#     nii_data = nii_data.reshape((96, 112))
#     plt.subplot(6, 10, fig_indx, xticks=[], yticks=[])
#     plt.imshow(nii_data[:, :], cmap='gray')
#     fig_indx += 1
# plt.savefig("model_found_bad_on_2d.png")
# plt.close()

print (real_outcome)
print (predict)
#print (sum(predict == real_y_for_anomaly))

# fig = plt.figure()
# fig_indx = 0
# for i in range(predict_good.shape[0]):
#     nii_data = predict_good[i]
#     nii_data = nii_data.reshape((96, 112))
#     plt.subplot(6, 10, fig_indx, xticks=[], yticks=[])
#     plt.imshow(nii_data[:, :], cmap='gray')
#     fig_indx += 1
# plt.savefig("model_found_good_on_2d.png")
# plt.close()
#
# joblib.dump(clf, brain_anomaly_model_path)



