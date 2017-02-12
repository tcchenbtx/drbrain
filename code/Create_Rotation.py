import os
from utils import process_nii
from utils import go_array
import numpy as np
import h5py
from PIL import Image
import nibabel as nib
import scipy.ndimage as snd
from matplotlib import pyplot as plt
from random import shuffle

# path
base_path = os.path.abspath(os.path.dirname(__file__))
base_path = os.path.join(base_path, "..")
raw_data_path = os.path.join(base_path, "raw_data")
code_path = os.path.join(base_path, "code")
log_path = os.path.join(code_path, "log")
preprocess_data_path = os.path.join(base_path, "preprocess_data")

# path to the Makefile
make_file_path = os.path.join(code_path, "Makefile")

# path to where to save the post-flirt nii
AD_afterflirt_path = os.path.join(preprocess_data_path, "after_flirt", "AD")
Normal_afterflirt_path = os.path.join(preprocess_data_path, "after_flirt", "Normal")
PD_afterflirt_path = os.path.join(preprocess_data_path, "after_flirt", "PD")

# get nii paths for subject unique

AD_nii_files = process_nii.get_nii(AD_afterflirt_path)
(unique_AD_subj, unique_AD_nii_files) = go_array.unique_sub_files(AD_nii_files)
Normal_nii_files = process_nii.get_nii(Normal_afterflirt_path)
(unique_Normal_subj, unique_Normal_nii_files) = go_array.unique_sub_files(Normal_nii_files)
PD_nii_files = process_nii.get_nii(PD_afterflirt_path)
(unique_PD_subj, unique_PD_nii_files) = go_array.unique_sub_files(PD_nii_files)

unique_data_dict = {"AD": unique_AD_nii_files, "Normal": unique_Normal_nii_files, "PD": unique_PD_nii_files}

############ functions for get randomized train valid test
def split_set(nii_dict):
    train = {}
    valid = {}
    test = {}
    for i in nii_dict.keys():
        nii_list = nii_dict[i]
        print(nii_list)
        shuffle(nii_list)
        print(nii_list)
        print(nii_list)
        train_to = int(len(nii_list) * 0.6)
        valid_to = int(len(nii_list) * 0.8)
        train["%s" % i] = nii_list[:train_to]
        valid["%s" % i] = nii_list[train_to:valid_to]
        test["%s" % i] = nii_list[valid_to:]
    return (train, valid, test)

############ split to get train, valid and test sets

(train_rotate_dict, valid_rotate_dict, test_rotate_dict) = split_set(unique_data_dict)

############ functions for making rotate images

def normalization_data (array_1d):
    out_mean = np.mean(array_1d)
    out_std = np.std(array_1d)
    output = (array_1d - out_mean)/out_std
    print("mean:")
    print(out_mean)
    return output

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

def output_h5(gopath, id, xinput, yinput, y_numinput):
    file_path = os.path.join(gopath, "%s.h5" % id)
    with h5py.File(file_path, 'w') as hf:
        hf.create_dataset("X", data=xinput)
        hf.create_dataset("Y", data=yinput)
        hf.create_dataset("Y_num", data=y_numinput)
    print ("%s data save at %s" % (id, file_path))
    return file_path

# need to input specific set for rotation for train, validate, test
def nii_to_1d_tf_no_mci_rotate(nii_dict, h5_path, nameit):
    # parameter setting
    y_label = {"Normal": 1, "AD": 2, "PD":3}
    npad = ((3, 2), (3, 0), (3, 2))
    rotate_degrees = [-15, -9, -6, -3, 3, 6, 9, 15]
    smooth = 1 # gausian smooth

    all_data_num = 0
    for i in nii_dict.keys():
        all_data_num += len(nii_dict[i])
    all_data_num_rotate = all_data_num * (len(rotate_degrees) + 1)

    # prepare output
    output_x = np.zeros((all_data_num_rotate, 96, 112, 96, 1), dtype="float32")
    print(output_x.shape)
    class_y_array = []

    data_count = 0
    print("input dictionary keys: %s" % str(nii_dict.keys()))
    go_over_list = []
    for i in nii_dict.keys():
        sub_files_list = nii_dict[i]
        for j in sub_files_list:
            go_over_list.append(j)
            print(go_over_list)
            nii_file = nib.load(j)
            nii_data = nii_file.get_data()
            print(nii_data.shape) # should be 91 x 109 x 91

            # pad to 96 x 112 x 96
            nii_data_pad = np.pad(nii_data, pad_width=npad, mode='constant', constant_values=0)

            # original : normalization + smooth
            nii_flat = np.ravel(nii_data_pad)
            nii_out = normalization_data(nii_flat)
            nii_out = snd.gaussian_filter(nii_out, smooth)
            output_x[data_count, :, :, :, :] = nii_out.astype("float32").reshape(1,96,112,96,1)
            print(y_label[i])
            class_y_array.append(y_label[i])
            data_count += 1


            # dimension 1
            #for k in rotate_degrees:
            #    rotate_nii = np.zeros((96,112,96), dtype='float32')
            #    for layer in range(nii_data_pad.shape[0]):
            #        rotate_layer = np.array(Image.fromarray(nii_data_pad[layer,:,:]).rotate(k, resample=Image.BICUBIC))
            #        rotate_nii[layer, :, :] = rotate_layer
            #    nii_flat = np.ravel(rotate_nii)
            #    nii_out = normalization_data(nii_flat)
            #    nii_out = snd.gaussian_filter(nii_out, smooth)
            #    output_x[data_count, :, :, :, :] = nii_out.astype("float32").reshape(1,96,112,96,1)
            #    print(y_label[i])
            #    class_y_array.append(y_label[i])
            #    data_count += 1
            # dimension 2
            #for k in rotate_degrees:
            #    rotate_nii = np.zeros((96,112,96), dtype='float32')
            #    for layer in range(nii_data_pad.shape[1]):
            #        rotate_layer = np.array(Image.fromarray(nii_data_pad[:,layer,:]).rotate(k, resample=Image.BICUBIC))
            #        rotate_nii[:, layer, :] = rotate_layer
            #    nii_flat = np.ravel(rotate_nii)
            #    nii_out = normalization_data(nii_flat)
            #    nii_out = snd.gaussian_filter(nii_out, smooth)
            #    output_x[data_count, :, :, :, :] = nii_out.astype("float32").reshape(1,96,112,96,1)
            #    print(y_label[i])
            #    class_y_array.append(y_label[i])
            #    data_count += 1
            # dimension 3
            for k in rotate_degrees:
                rotate_nii = np.zeros((96,112,96), dtype='float32')
                for layer in range(nii_data_pad.shape[2]):
                    rotate_layer = np.array(Image.fromarray(nii_data_pad[:,:,layer]).rotate(k, resample=Image.BICUBIC))
                    rotate_nii[:, :, layer] = rotate_layer
                nii_flat = np.ravel(rotate_nii)
                nii_out = normalization_data(nii_flat)
                nii_out = snd.gaussian_filter(nii_out, smooth)
                output_x[data_count, :, :, :, :] = nii_out.astype("float32").reshape(1,96,112,96,1)
                print(y_label[i])
                class_y_array.append(y_label[i])
                data_count += 1

    #####

    print("output_x shpae:")
    print(output_x.shape)
    print("class_y_array shape:")
    print(len(class_y_array))

    output_y_onehot = label_convert_no_mci(class_y_array).astype("float32")
    class_y_array = np.array(class_y_array)

    print("output_y_onehot shape:")
    print(output_y_onehot.shape)

    # randomize matrix for both x and y

    random_order = np.random.permutation(output_x.shape[0]).argsort()
    print (output_x.shape[0])

    final_output_x = np.take(output_x, random_order, axis=0)
    final_output_y_onehot = np.take(output_y_onehot, random_order, axis=0)
    final_output_y_number = np.take(class_y_array, random_order, axis=0)


    output_file_path = output_h5(h5_path, nameit, final_output_x, final_output_y_onehot, final_output_y_number)

    print(final_output_x.dtype)
    print(final_output_y_onehot.dtype)
    print(final_output_x.shape)
    print(final_output_y_onehot.shape)
    print(final_output_y_onehot[:5, :])
    print(final_output_y_number[:5])


#     ### check image check point
#     if tf:
#         first_hundred = final_output_x[:100, :, :, :, :]
#         fig_indx = 1
#         for i in range(100):
#             plt.subplot(4, 25, fig_indx, xticks=[], yticks=[])
#             go_reshape = first_hundred[i, :, :, :, :].reshape(96, 112, 96)
#             plt.imshow(go_reshape[:,:, 45], cmap='gray')
#             fig_indx += 1
#         plt.savefig("tf_format_check_no_mci.png")
#     ##########


    return (final_output_x, final_output_y_onehot, output_file_path)

############### get rotate image and save as h5 files
## train set

(train_x, train_y, train_path) = nii_to_1d_tf_no_mci_rotate(train_rotate_dict, preprocess_data_path, "train_rotate")
print(train_x.shape)
print(train_y.shape)
print(np.sum(train_x[0,:,:,:,:]))
print(np.std(train_x[0,:,:,:,:]))
print(train_y[:5,:])

del train_x
del train_y

(valid_x, valid_y, valid_path) = nii_to_1d_tf_no_mci_rotate(valid_rotate_dict, preprocess_data_path, "valid_rotate")
print(valid_x.shape)
print(valid_y.shape)
print(np.sum(valid_x[0,:,:,:,:]))
print(np.std(valid_x[0,:,:,:,:]))
print(valid_y[:5,:])

del valid_x
del valid_y

(test_x, test_y, test_path) = nii_to_1d_tf_no_mci_rotate(test_rotate_dict, preprocess_data_path, "test_rotate")
print(test_x.shape)
print(test_y.shape)
print(np.sum(test_x[0,:,:,:,:]))
print(np.std(test_x[0,:,:,:,:]))
print(test_y[:5,:])

del test_x
del test_y


