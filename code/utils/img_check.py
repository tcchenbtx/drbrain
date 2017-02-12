from matplotlib import pyplot as plt
import nibabel as nib
import h5py
import random

# visualize subject x 1D data
def view_data(h5_file_not_tf, fig_num, checkname):
    with h5py.File(h5_file_not_tf, 'r') as hf:
        X = hf.get('X')
        pick = random.sample(range(X.shape[0]), fig_num)

        fig_indx = 0
        f, ax = plt.subplots(1, fig_num, sharex=True, sharey=True)
        target_files = [X[each,:,:,:,:] for each in pick]
        for each_nii in target_files:
            data = each_nii.reshape(96,112,96)
            middle_last = int(96/2)
            ax[fig_indx].imshow(data[:, :, middle_last], cmap='gray')
            fig_indx += 1
    plt.savefig("%s.png" % checkname)


# 3D image check: visualize middle image on z axis globally
def check_all(nii_path_list, checkname):
    plot_num = len(nii_path_list)
    print ("Should see %d images in %s" % (plot_num, checkname))
    plot_high = int(plot_num/25) + 1
    fig = plt.figure(figsize=[200, 200])
    fig_indx = 0
    for i in nii_path_list:
        print(fig_indx)
        print(i)
        nii_load = nib.load(i)
        nii_data = nii_load.get_data()
        last_middle = int(nii_data.shape[2]/2)
        plt.subplot(plot_high, 25, fig_indx, xticks=[], yticks=[])
        plt.imshow(nii_data[:,:, last_middle], cmap='gray')
        plt.title("%s" % str(fig_indx), fontsize=10, weight='bold')
        fig_indx += 1
    plt.savefig("%s_check.png" % checkname)
    plt.close()

