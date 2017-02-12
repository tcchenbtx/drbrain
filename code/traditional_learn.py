import os
import h5py
from sklearn.metrics import roc_auc_score


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


with h5py.File(unique_matrix_path, 'r') as hf:
    X = hf.get('X')
    Y = hf.get('Y')
    Y_num = hf.get('Y_num')
    print(X.shape)
    print(Y.shape)
    print(len(Y_num))

    train_subset_X = X[:645, :]
    train_subset_Y = Y_num[:645]
    valid_subset_X = X[645:860,:]
    valid_subset_Y = Y_num[645:860]
    test_subset_X = X[860:,:]
    test_subset_Y = Y_num[860:]



# centers = [[1, 1], [-1, -1], [1, -1]]
#
#
#
# fig = plt.figure(1, figsize=(4, 3))
# plt.clf()
# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
#
# plt.cla()
# pca = PCA(n_components=3)
# pca.fit(X)
#
# X = pca.transform(X)
#
#
# for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
#     ax.text3D(X[y == label, 0].mean(),
#               X[y == label, 1].mean() + 1.5,
#               X[y == label, 2].mean(), name,
#               horizontalalignment='center',
#               bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.datasets import make_moons, make_circles, make_classification  # Use my own dataset
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# h = .02  # step size in the mesh

# names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
#          "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
#          "Naive Bayes", "QDA"]
#
# classifiers = [
#     KNeighborsClassifier(4),
#     SVC(kernel="linear", C=0.025),
#     SVC(gamma=2, C=1),
#     GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
#     DecisionTreeClassifier(max_depth=5),
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#     MLPClassifier(alpha=1),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     QuadraticDiscriminantAnalysis()]

# X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
#                            random_state=1, n_clusters_per_class=1)
# rng = np.random.RandomState(2)
# X += 2 * rng.uniform(size=X.shape)
# linearly_separable = (X, y)

# datasets = [make_moons(noise=0.3, random_state=0),
#             make_circles(noise=0.2, factor=0.5, random_state=1),
#             linearly_separable
#             ]

# figure = plt.figure(figsize=(6, 8))
# i = 1
# iterate over datasets
# for ds_cnt, ds in enumerate(datasets):
#     # preprocess dataset, split into training and test part
#     X, y = ds
#     X = StandardScaler().fit_transform(X)
#     X_train, X_test, y_train, y_test = \
#         train_test_split(X, y, test_size=.4, random_state=42)
#
#     x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
#     y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))
#
#     # just plot the dataset first
#     cm = plt.cm.RdBu
#     cm_bright = ListedColormap(['#FF0000', '#0000FF'])
#     ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
#     if ds_cnt == 0:
#         ax.set_title("Input data")
#     # Plot the training points
#     ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
#     # and testing points
#     ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
#     ax.set_xlim(xx.min(), xx.max())
#     ax.set_ylim(yy.min(), yy.max())
#     ax.set_xticks(())
#     ax.set_yticks(())
#     i += 1

# iterate over classifiers
#for name, clf in zip(names, classifiers):
    # ax = plt.subplot(1, len(classifiers) + 1, i)
#    clf.fit(train_subset_X, train_subset_Y)
#    score = clf.score(valid_subset_X, valid_subset_Y)

#    print ("%s_score:" % name)
#    print (score)

# for name, clf in zip(names, classifiers):
#     # ax = plt.subplot(1, len(classifiers) + 1, i)
#     clf.fit(train_subset_X, train_subset_Y)
#     score = clf.score(valid_subset_X, valid_subset_Y)
#
#     print ("%s_score:" % name)
#     print (score)
#
#

#### focus on linear SVM
clf = SVC(kernel="linear", C=0.025)
clf.fit(train_subset_X, train_subset_Y)
#
score_train = clf.score(train_subset_X, train_subset_Y)
print("train_score")
print(score_train)
score_valid = clf.score(valid_subset_X, valid_subset_Y)
print("valid_score")
print(score_valid)
score_test = clf.score(test_subset_X, test_subset_Y)
print("test_score")
print(score_test)

y_predict = clf.predict(valid_subset_X)
myauc = roc_auc_score(valid_subset_Y, y_predict)
print("AUC")
print(myauc)

## grid search
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# parameters = {'kernel':('linear'), 'C':[0.01, 0.1, 1, 10, 100]}
# svr = svm.SVC()
# clf = GridSearchCV(svr, parameters)
# clf.fit(train_subset_X, train_subset_Y)


# Set the parameters by cross-validation
#tuned_parameters = [{'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100]}]

#scores = ['precision', 'recall']

#for score in scores:
#    print("# Tuning hyper-parameters for %s" % score)
#    print()

#    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring='%s_macro' % score)
#    clf.fit(train_subset_X, train_subset_Y)

#    print("Best parameters set found on development set:")
#    print()
#    print(clf.best_params_)
#    print()
#    print("Grid scores on development set:")
#    print()
#    means = clf.cv_results_['mean_test_score']
#    stds = clf.cv_results_['std_test_score']
#    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#        print("%0.3f (+/-%0.03f) for %r"
#              % (mean, std * 2, params))

#    print()

#    print("Detailed classification report:")
#    print()
#    print("The model is trained on the full development set.")
#    print("The scores are computed on the full evaluation set.")
#    print()
#    y_true, y_pred = valid_subset_Y, clf.predict(valid_subset_X)
#    print(classification_report(y_true, y_pred))
#    print()
#    y_true, y_pred = test_subset_Y, clf.predict(test_subset_X)
#    print(classification_report(y_true, y_pred))







#     # Plot the decision boundary. For that, we will assign a color to each
#     # point in the mesh [x_min, x_max]x[y_min, y_max].
#     if hasattr(clf, "decision_function"):
#         Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
#     else:
#         Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
#
#     # Put the result into a color plot
#     Z = Z.reshape(xx.shape)
#     ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
#
#     # Plot also the training points
#     ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
#     # and testing points
#     ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
#                    alpha=0.6)
#
#     ax.set_xlim(xx.min(), xx.max())
#     ax.set_ylim(yy.min(), yy.max())
#     ax.set_xticks(())
#     ax.set_yticks(())
#     # if ds_cnt == 0:
#     #     ax.set_title(name)
#     ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
#                 size=15, horizontalalignment='right')
#     # i += 1
#
#
# plt.save("fit_tradition.png")
# plt.tight_layout()
# plt.show()
