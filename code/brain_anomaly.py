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

# where to save model
brain_anomaly_model_path = os.path.join(models_path, "brain_anomaly.pkl")

# create bad data matrix
# Bad_afterflirt_path = os.path.join(preprocess_data_path, "after_flirt", "Bad")
# Bad_nii_files = process_nii.get_nii(Bad_afterflirt_path)
# bad_data_dict = {"Bad": Bad_nii_files}
#
# bad_data_matrix = go_array.nii_to_1d_simple(bad_data_dict, bad_matrix_path, normalization=True, smooth=1, tf=False)
#
# print("bad matrix:")
# print(bad_data_matrix.dtype)
# print(bad_data_matrix.shape)

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

### PCA plot
target_names = ['Good', 'Bad']

# pca = PCA(n_components=2)
# X_r = pca.fit(good_X).transform(X)

#lda = LinearDiscriminantAnalysis(n_components=2)
#X_r2 = lda.fit(X, y_label_list).transform(X)

# Percentage of variance explained for each components
# print('explained variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))

#plt.figure()
#colors = ['navy', 'magenta']
#lw = 1


## Good = 0, Bad = 1
#print("X_r2 shape:")
#print(X_r2.shape)

#for color, i, target_name in zip(colors, [0, 1], target_names):
#    plt.scatter(X_r2[y_label == i, 0], X_r2[y_label == i, 1], color=color, alpha=.8, lw=lw, label=target_name, s=10)
#    plt.legend(loc='best', shadow=False, scatterpoints=1)

#plt.title('Abnormal Data')
#plt.xlabel('LD1')
#plt.ylabel('LD2')
#plt.savefig("Abnormal_data_LDA.png")



#### abnormaly
# # train with X_good
#
X_train = good_X[:1025,:]
print("X_train:")
print(X_train.shape)
X_test = np.vstack((good_X[1025:, :], bad_X))
print("X_test:")
print(X_test.shape)
#
real_outcome = ([1] * good_X[1025:,:].shape[0]) + ([0] * bad_X.shape[0])
real_outcome = np.array(real_outcome)
print(real_outcome.shape)
# #
print("X_train")
print(X_train.shape)
print("X_test")
print(X_test.shape)
#
#

rng = np.random.RandomState(42)
brain_anomaly = IsolationForest(max_samples='auto', random_state=rng)
brain_anomaly.fit(X_train)


# 1 is good
# -1 is bad
predict = brain_anomaly.predict(X_test)
#
predict_array = np.array(predict)
predict_bad = X_test[predict_array == -1]
predict_good = X_test[predict_array == 1]
#
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


#
#
print (predict)
#print (sum(predict == real_y_for_anomaly))

## save model

joblib.dump(brain_anomaly, brain_anomaly_model_path)



## tran with PCA to 10000

# pca_anomaly = PCA(n_components=10000)
# pca_anomaly.fit(good_X)
#
# X_train_pca = pca_anomaly.transform(X_train)
#
# rng = np.random.RandomState(42)
# clf_pca = IsolationForest(max_samples='auto', random_state=rng)
# clf_pca.fit(X_train_pca)
#
# X_test_pca = pca_anomaly.transform(X_test)
# predict = clf_pca.predict(X_test_pca)
#
# print(predict)
#print(sum(predict == real_y_for_anomaly))



