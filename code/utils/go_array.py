import numpy as np
import nibabel as nib
import scipy.ndimage as snd
import h5py
import os
import re
from matplotlib import pyplot as plt

# for reproducibility
np.random.seed(49979687) # a big prime number

# get unique subject
def unique_sub_files(list_of_path):
    subid_pattern = re.compile(r'/([A-Za-z0-9_]*)_[A-Za-z0-9]*\.nii$|/([A-Za-z0-9_]*)_[A-Za-z0-9]*\.nii.gz$')
    sub_list = []
    output_list = []
    for i in list_of_path:
        check_subid = re.findall(subid_pattern, i)
        if check_subid[0] in sub_list:
            continue
        else:
            sub_list.append(check_subid[0])
            output_list.append(i)
    print("total unique subject number = %d" % len(sub_list))
    print("unique data number = %d" % len(output_list))
    return (sub_list, output_list)


# normalize 1 D array with (x-mean)/std
def normalization_data (array_1d):
    out_mean = np.mean(array_1d)
    out_std = np.std(array_1d)
    output = (array_1d - out_mean)/out_std
    print("mean:")
    print(out_mean)

    return output

# smooth data
def data_smooth(input_subj_per_row, g_num):
    output = np.zeros_like(input_subj_per_row)
    for i in range(0, input_subj_per_row.shape[0]):
        output[i, :] = snd.gaussian_filter(input_subj_per_row[i, :], g_num)
    return output

# onehot for 4 categories
def label_convert (y_array):
    """
    1-normal: [1,0,0,0]
    2-MCI:[0,1,0,0]
    3-AD:[0,0,1,0]
    4-PD:[0,0,0,1]

    """
    # label for each condition
    normal = np.array([1,0,0,0])
    mci = np.array([0,1,0,0])
    ad = np.array([0,0,1,0])
    pd = np.array([0,0,0,1])
    assign_matrix = [normal, mci, ad, pd]
    output = np.zeros((len(y_array),4))
    ind = 0
    print("y_array length: %d" % len(y_array))
    for i in y_array:
        # print(i)
        # print(assign_matrix[i-1])
        output[ind, :] = assign_matrix[i-1]
        ind += 1
    return output

# onehot for 3 categories
def label_convert_no_mci (y_array):
    """
    1-normal: [1,0,0]
    2-AD:[0,1,0]
    3-PD:[0,0,1]

    """
    # label for each condition
    normal = np.array([1,0,0])
    ad = np.array([0,1,0])
    pd = np.array([0,0,1])
    assign_matrix = [normal, ad, pd]
    output = np.zeros((len(y_array), 3))
    ind = 0
    print("y_array length: %d" % len(y_array))
    for i in y_array:
        # print(i)
        # print(assign_matrix[i-1])
        output[ind, :] = assign_matrix[i-1]
        ind += 1
    return output

# code to output h5 file
def output_h5(gopath, id, xinput, yinput, y_numinput):
    file_path = os.path.join(gopath, "%s.h5" % id)
    with h5py.File(file_path, 'w') as hf:
        hf.create_dataset("X", data=xinput)
        hf.create_dataset("Y", data=yinput)
        hf.create_dataset("Y_num", data=y_numinput)
    print ("%s data save at %s" % (id, file_path))
    return file_path

# convert nii 3D images to 1 D vector (4 categories)
def nii_to_1d(nii_dict, h5_path, nameit, normalization=True, smooth=None, tf=False):
    y_label = {"Normal": 1, "MCI":2, "AD": 3, "PD":4}
    # padding info to make the output 96 x 112  x 96
    npad = ((3, 2), (3, 0), (3, 2))

    all_data_num = 0
    for i in nii_dict.keys():
        all_data_num += len(nii_dict[i])
    output_x = np.zeros((all_data_num, 96*112*96), dtype="float32")
    class_y_array = []

    # go over all data for data processing
    data_count = 0
    print("input dictionary keys: %s" % str(nii_dict.keys()))
    go_over_list = []
    for i in nii_dict.keys():
        sub_files_list = nii_dict[i]
        for j in sub_files_list:
            go_over_list.append(j)
            nii_file = nib.load(j)
            nii_data = nii_file.get_data()
            print(nii_data.shape)

            nii_data_pad = np.pad(nii_data, pad_width=npad, mode='constant', constant_values=0)
            nii_flat = np.ravel(nii_data_pad)


            if normalization:
                nii_out = normalization_data(nii_flat)
                # check if nan in array
                if np.isnan(np.sum(nii_out)):
                    print("%s will lead to nan!!!!" % j)
            else:
                nii_out = nii_flat

            if smooth != None:
                nii_out = snd.gaussian_filter(nii_out, smooth)
            else:
                nii_out = nii_out

            output_x[data_count, :] = nii_out.astype("float32")


            print(y_label[i])
            class_y_array.append(y_label[i])

            data_count += 1

    # check point
    print("output_x shpae:")
    print(output_x.shape)
    print("class_y_array shape:")
    print(len(class_y_array))

    # onehot conversion
    output_y_onehot = label_convert(class_y_array).astype("float32")

    print("output_y_onehot shape:")
    print(output_y_onehot.shape)

    # randomize matrix for both x and y
    random_order = np.random.permutation(output_x.shape[0]).argsort()
    print (output_x.shape[0])

    final_output_x = np.take(output_x, random_order, axis=0)
    final_output_y_onehot = np.take(output_y_onehot, random_order, axis=0)
    final_output_y_number = np.take(class_y_array, random_order, axis=0)

    # if output tensorflow shape
    if tf == True:  # output n x 96 x 112 x 96 x 1
        tf_output_format = np.zeros((final_output_x.shape[0], 96, 112, 96, 1))
        for i in range(final_output_x.shape[0]):
            tf_output_format[i, :, :, :, :] = final_output_x[i, :].reshape(1, 96, 112, 96, 1)
        final_output_x = tf_output_format.astype("float32")

    # save as h5 file
    output_file_path = output_h5(h5_path, nameit, final_output_x, final_output_y_onehot, final_output_y_number)

    # check dtype and shape
    print(final_output_x.dtype)
    print(final_output_y_onehot.dtype)
    print(final_output_x.shape)
    print(final_output_y_onehot.shape)
    print(final_output_y_onehot[:5, :])
    print(final_output_y_number[:5])

    # plot images for visual check
    ### check image check point
    if tf:
        first_hundred = final_output_x[:100, :, :, :, :]
        fig_indx = 1
        for i in range(100):
            plt.subplot(4, 25, fig_indx, xticks=[], yticks=[])
            go_reshape = first_hundred[i, :, :, :, :].reshape(96, 112, 96)
            plt.imshow(go_reshape[:,:, 45], cmap='gray')
            fig_indx += 1
        plt.savefig("tf_format_check.png")
    ###
    return (final_output_x, final_output_y_onehot, output_file_path)


# convert nii 3D images to 1 D vector (for "Bad" images)
def nii_to_1d_simple(nii_list, h5_path_name, normalization=True, smooth=None, tf=False):  # for processing Bad data
    # padding info
    npad = ((3, 2), (3, 0), (3, 2))

    all_data_num = len(nii_list)
    output_x = np.zeros((all_data_num, 96*112*96), dtype="float32")
    # class_y_array = []

    # go over all nii files
    data_count = 0
    for i in nii_list:
        nii_file = nib.load(i)
        nii_data = nii_file.get_data()
        print(nii_data.shape)

        nii_data_pad = np.pad(nii_data, pad_width=npad, mode='constant', constant_values=0)
        nii_flat = np.ravel(nii_data_pad)


        if normalization:
            nii_out = normalization_data(nii_flat)
            # check if nan in array
            if np.isnan(np.sum(nii_out)):
                print("nan!!!!")
        else:
            nii_out = nii_flat

        if smooth != None:
            nii_out = snd.gaussian_filter(nii_out, smooth)
        else:
            nii_out = nii_out

        output_x[data_count, :] = nii_out.astype("float32")
        data_count += 1


    # if output tensorflow shape
    if tf == True:  # output n x 96 x 112 x 96 x 1
        tf_output_format = np.zeros((output_x.shape[0], 96, 112, 96, 1))
        for i in range(output_x.shape[0]):
            tf_output_format[i, :, :, :, :] = output_x[i, :].reshape(1, 96, 112, 96, 1)
        final_output_x = tf_output_format.astype("float32")
    else:
        final_output_x = output_x

    # output h5 file
    with h5py.File(h5_path_name, 'w') as hf:
        hf.create_dataset("X", data=final_output_x)

    print(final_output_x.dtype)
    print(final_output_x.shape)

    ### check image check point

    first_ten = final_output_x[:10, :]
    fig_indx = 1
    for i in range(10):
        plt.subplot(2, 5, fig_indx, xticks=[], yticks=[])
        go_reshape = first_ten[i, :].reshape((96, 112, 96))
        plt.imshow(go_reshape[:, :, 45], cmap='gray')
        fig_indx += 1
    plt.savefig("bad_img_check.png")
    ##########

    return final_output_x


# convert nii 3D images to 1 D vector (3 categories)
def nii_to_1d_no_mci(nii_dict, h5_path, nameit, normalization=True, smooth=None, tf=False):
    # how to onehot coding
    y_label = {"Normal": 1, "AD": 2, "PD":3}
    # padding info
    npad = ((3, 2), (3, 0), (3, 2))

    all_data_num = 0
    for i in nii_dict.keys():
        all_data_num += len(nii_dict[i])
    output_x = np.zeros((all_data_num, 96*112*96), dtype="float32")
    class_y_array = []

    # go through all nii files
    data_count = 0
    print("input dictionary keys: %s" % str(nii_dict.keys()))
    go_over_list = []
    for i in nii_dict.keys():
        sub_files_list = nii_dict[i]
        for j in sub_files_list:
            go_over_list.append(j)
            nii_file = nib.load(j)
            nii_data = nii_file.get_data()
            print(nii_data.shape)

            nii_data_pad = np.pad(nii_data, pad_width=npad, mode='constant', constant_values=0)
            nii_flat = np.ravel(nii_data_pad)


            if normalization:
                nii_out = normalization_data(nii_flat)
                # check if nan in array
                if np.isnan(np.sum(nii_out)):
                    print("%s will lead to nan!!!!" % j)
            else:
                nii_out = nii_flat

            if smooth != None:
                nii_out = snd.gaussian_filter(nii_out, smooth)
            else:
                nii_out = nii_out

            output_x[data_count, :] = nii_out.astype("float32")


            print(y_label[i])
            class_y_array.append(y_label[i])

            data_count += 1


    print("output_x shpae:")
    print(output_x.shape)
    print("class_y_array shape:")
    print(len(class_y_array))

    # onehot coding
    output_y_onehot = label_convert_no_mci(class_y_array).astype("float32")

    print("output_y_onehot shape:")
    print(output_y_onehot.shape)

    # randomize matrix for both x and y
    random_order = np.random.permutation(output_x.shape[0]).argsort()
    print (output_x.shape[0])

    final_output_x = np.take(output_x, random_order, axis=0)
    final_output_y_onehot = np.take(output_y_onehot, random_order, axis=0)
    final_output_y_number = np.take(class_y_array, random_order, axis=0)

    # if output tensorflow shape
    if tf == True:  # output n x 96 x 112 x 96 x 1
        tf_output_format = np.zeros((final_output_x.shape[0], 96, 112, 96, 1))
        for i in range(final_output_x.shape[0]):
            tf_output_format[i, :, :, :, :] = final_output_x[i, :].reshape(1, 96, 112, 96, 1)
        final_output_x = tf_output_format.astype("float32")

    # output as h5 file
    output_file_path = output_h5(h5_path, nameit, final_output_x, final_output_y_onehot, final_output_y_number)

    print(final_output_x.dtype)
    print(final_output_y_onehot.dtype)
    print(final_output_x.shape)
    print(final_output_y_onehot.shape)
    print(final_output_y_onehot[:5, :])
    print(final_output_y_number[:5])

    # plot images for visual check
    ### check image check point
    if tf:
        first_hundred = final_output_x[:100, :, :, :, :]
        fig_indx = 1
        for i in range(100):
            plt.subplot(4, 25, fig_indx, xticks=[], yticks=[])
            go_reshape = first_hundred[i, :, :, :, :].reshape(96, 112, 96)
            plt.imshow(go_reshape[:,:, 45], cmap='gray')
            fig_indx += 1
        plt.savefig("tf_format_check_no_mci.png")
    ##########


    return (final_output_x, final_output_y_onehot, output_file_path)