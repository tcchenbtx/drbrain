import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#iris = datasets.load_iris()


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
    #
    # train_subset_X = X[:752, :]
    # train_subset_Y = Y_num[:752]
    # valid_subset_X = X[752:,:]
    # valid_subset_Y = Y_num[752:]


# 1-normal: [1,0,0,0]
#     2-MCI:[0,1,0,0]
#     3-AD:[0,0,1,0]
#     4-PD:[0,0,0,1]




# X = iris.data
# y = iris.target
    target_names = ['Normal', 'MCI', 'AD', 'PD']




    # pca = PCA(n_components=2)
    # X_r = pca.fit(X).transform(X)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, Y_num).transform(X)

# Percentage of variance explained for each components
#     print('explained variance ratio (first two components): %s'
#           % str(pca.explained_variance_ratio_))

    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange', 'magenta']
    lw = 1

    Y_num = np.array(Y_num)

    # for color, i, target_name in zip(colors, [1, 2, 3, 4], target_names):
    #     plt.scatter(X_r[Y_num == i, 0], X_r[Y_num == i, 1], color=color, alpha=.8, lw=lw,
    #                 label=target_name, s=10)
    # plt.legend(loc='best', shadow=False, scatterpoints=1)
    # plt.title('Component Space')
    # plt.xlabel('PC1')
    # plt.ylabel('PC2')
    #
    # plt.savefig("dimension_reduction_PCA.png")

    plt.figure()
    for color, i, target_name in zip(colors, [1, 2, 3, 4], target_names):
        plt.scatter(X_r2[Y_num == i, 0], X_r2[Y_num == i, 1], alpha=.8, color=color,
                    label=target_name, s=10)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA')

    plt.savefig("dimention_reduction_LDA.png")
