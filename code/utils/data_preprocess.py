import numpy as np
import scipy.ndimage as snd
import h5py


# have x to be one subject per row
def X_subject_per_row(inputdata, expected_unit_size):
    subject_num = int(inputdata.shape[0]/expected_unit_size[0])
    num_voxel_per_subj = reduce(lambda x, y: x*y, expected_unit_size)
    output = np.zeros((subject_num, num_voxel_per_subj))
    print (output.shape)
    ind = 0
    for i in range(0, inputdata.shape[0], expected_unit_size[0]):
        output[ind,] = np.ravel(inputdata[i:i+expected_unit_size[0],:,:])
        ind += 1
    return output

# have x to be 1 x num_sample x 3D(channels x rows x columns)
def X_subj_3D(input_sub_per_row, expected_unit_size):
    """
    th dim_ordering = samples x channels x dim1, dim2, dim3
    => need to use subject_num x 1 x 62 x 96 x 96 (the expected_unit_seize)

    """
    subject_num = input_sub_per_row.shape[0]
    output_shape = [subject_num] + [1] + expected_unit_size
    output = input_sub_per_row.reshape(np.array(output_shape))

    # batch_size = np.array(expected_unit_size)
    # output = np.zeros((output_shape))
    print (output_shape)
    # for i in range(subject_num):
    #     output[0, i, :, :, :] = input_sub_per_row[i, :].reshape(batch_size)
    return output

# smoothing figure
def data_smooth(input_subj_per_row):
    output = np.zeros_like(input_subj_per_row)
    for i in range(0, input_subj_per_row.shape[0]):
        output[i, :] = snd.gaussian_filter(input_subj_per_row[i, :], 1)
    return output



# have y to be one subject per row
def Y_subject_per_row(total_y, expected_unit_size):
    output = []
    for i in range(0, total_y.shape[0], expected_unit_size[0]):
        output.append(total_y[i])
    return output

# scale 1 D array with (x-mean)/std
def scale_data (array_1d):
    out_mean = np.mean(array_1d)
    out_std = np.std(array_1d)
    output = (array_1d - out_mean)/out_std
    return output

# scale group_x_subj_per_row
def scale_X (group_x_per_subj):
    output = np.zeros_like(group_x_per_subj)
    for i in range(group_x_per_subj.shape[0]):
        output[i,:] = scale_data(group_x_per_subj[i,:])
    return output

# convert y label 1, 2, 3 to [1,0,0], [0,1,0], [0,0,1]
def label_convert (group_y_per_subj):
    """
    1-normal: [1,0,0]
    2-MCI:[0,1,0]
    3-AD:[0,0,1]

    """
    # label for each condition
    normal = np.array([1,0,0])
    mci = np.array([0,1,0])
    ad = np.array([0,0,1])
    assign_matrix = [normal, mci, ad]
    output = np.zeros((len(group_y_per_subj),3))
    ind = 0
    for i in group_y_per_subj:
        output[ind,:] = assign_matrix[i-1]
        ind += 1

    return output

# for batch training using fit_generator
def data_array_generator(xfile, yfile):
    while 1:
        with h5py.File(xfile, 'r') as h5_x, h5py.File(yfile, 'r') as h5_y:
            Xdata = h5_x.get('train_X_input')
            Ydata = h5_y.get('train_Y_input')
            for i in range(Xdata.shape[0]):
                print(i)
                x_array = np.array([Xdata[i, :, :, :, :]])
                y_array = np.array([Ydata[i, :]])
                yield (x_array, y_array)

# for inceptionv3_go
def inceptv3_data_generator(xfile, yfile, xinfo, yinfo):
    while 1:
        with h5py.File(xfile, 'r') as h5_x, h5py.File(yfile, 'r') as h5_y:
            Xdata = h5_x.get(xinfo)
            Ydata = h5_y.get(yinfo)
            for i in range(Xdata.shape[0]):
                x_array = Xdata[i, :, :, :, :].reshape(1,1,558,1024)
                x_array = x_array.astype('float32')
                y_array = Ydata[i, :]
                y_array = y_array.astype('float32')
                yield (x_array, y_array)



# for autoencoder_method
def fixed_data_array_generator(xfile, yfile, xinfo, yinfo):
    while 1:
        with h5py.File(xfile, 'r') as h5_x, h5py.File(yfile, 'r') as h5_y:
            Xdata = h5_x.get(xinfo)
            Ydata = h5_y.get(yinfo)
            x_array = np.zeros((1, 1, 64, 96, 96))
            for i in range(Xdata.shape[0]):
                print(i)
                x_array[:, :, 1:63, :, :] = Xdata[i, :, :, :, :]
                print(x_array.shape)
                y_array = Ydata[i, :]
                print(y_array.shape)
                yield (x_array, y_array)




def fixed_data_array_generator_autoencoder(xfile, info):
    while 1:
        with h5py.File(xfile, 'r') as h5_x:
            Xdata = h5_x.get(info)
            x_array = np.zeros((1, 1, 64, 96, 96))
            for i in range(Xdata.shape[0]):
                x_array[:, :, 1:63, :, :] = Xdata[i, :, :, :, :]
                x_array = x_array.astype('float32')
                # print(x_array.dtype)
                yield (x_array, x_array)


# for getting encoded figures
def data_for_encoder(xfile, info):
    while 1:
        with h5py.File(xfile, 'r') as h5_x:
            Xdata = h5_x.get(info)
            x_array = np.zeros((1, 1, 64, 96, 96))
            for i in range(Xdata.shape[0]):
                print (i)
                x_array[:, :, 1:63, :, :] = Xdata[i, :, :, :, :]
                yield x_array

# for inception v3 with tf setting
def inceptv3_data_generator_tf(xfile, yfile, xinfo, yinfo):
    while 1:
        with h5py.File(xfile, 'r') as h5_x, h5py.File(yfile, 'r') as h5_y:
            Xdata = h5_x.get(xinfo)
            Ydata = h5_y.get(yinfo)
            shuffle = np.random.permutation(Xdata.shape[0]).argsort()
            for i in shuffle:
                x_array = Xdata[i, :, :, :, :].reshape(1,558,1024,1)
                x_array = x_array.astype('float32')
                y_array = Ydata[i, :].reshape(1,3)
                y_array = y_array.astype('float32')
                yield (x_array, y_array)


# for inception v3 with tf setting
def inceptv3_data_generator_3d_tf(xfile, yfile, xinfo, yinfo):
    while 1:
        with h5py.File(xfile, 'r') as h5_x, h5py.File(yfile, 'r') as h5_y:
            Xdata = h5_x.get(xinfo)
            Ydata = h5_y.get(yinfo)
            shuffle = np.random.permutation(Xdata.shape[0]).argsort()
            for i in shuffle:
                x_array = Xdata[i, :, :, :, :].reshape(1,62, 96, 96,1)
                x_array = x_array.astype('float32')
                y_array = Ydata[i, :].reshape(1,3)
                y_array = y_array.astype('float32')
                yield (x_array, y_array)





# def generate_arrays_from_file(path):
#     while 1:
#         f = open(path)
#         for line in f:
#             # create numpy arrays of input data
#             # and labels, from each line in the file
#             x, y = process_line(line)
#             img = load_images(x)
#             yield (img, y)
#         f.close()
# #
# # model.fit_generator(generate_arrays_from_file('/my_file.txt'),
# #         samples_per_epoch=10000, nb_epoch=10)
