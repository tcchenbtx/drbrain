import os
import re
import random
import nibabel as nib
from matplotlib import pyplot as plt
import subprocess


# path
base_path = os.path.abspath(os.path.dirname(__file__))
base_path = os.path.join(base_path, "..")
raw_data_path = os.path.join(base_path, "raw_data")
code_path = os.path.join(base_path, "code")
utils_path = os.path.join(code_path, "utils")

# path to process dcm files
AD_raw_data_path = os.path.join(raw_data_path, "AD")
MCI_raw_data_path = os.path.join(raw_data_path, "MCI")
Normal_raw_data_path = os.path.join(raw_data_path, "Normal")

make_file_path = os.path.join(code_path, "Makefile")

def get_nii(gopath):
    file_pattern = re.compile(r'.*\.nii$|.*\.nii\.gz$')
    nii_file_list = []
    for root, dirs, files in os.walk(gopath):
        for file in files:
            if re.match(file_pattern, file):
                nii_file_list.append(os.path.join(root, file))
    return nii_file_list

def random_visualization(fig_num, file_dict, filename):
    group_num = len(file_dict)
    f, ax = plt.subplots(group_num, fig_num, sharex=True, sharey=True)
    plt.setp(ax, xticks=[], xticklabels=[], yticks=[])
    group_indx = 0
    sorted_keys = sorted(file_dict.keys())
    for i in sorted_keys:
        file_list = file_dict[i]
        fig_indx = 0
        pick = random.sample(range(len(file_list)), fig_num)
        target_files = [file_list[each] for each in pick]
        for each_nii in target_files:
            nii_file = nib.load(each_nii)
            nii_data = nii_file.get_data()
            if len(nii_data.shape) == 4:
                nii_data = nii_data.reshape(nii_data.shape[:-1])
            print(nii_data.shape)
            middle_last = int(nii_data.shape[2]/2)
            print(middle_last)
            ax[group_indx, fig_indx].imshow(nii_data[:, :, middle_last], cmap='gray')
            ax[group_indx, fig_indx].set_title("%s" % i)
            fig_indx += 1
        group_indx += 1
    plt.savefig("%s.png" % filename)
    return

def fix_orientation(id, gopath, outpath, makefile, logpath="./"):
   nii_list = get_nii(gopath)
   file_name_pattern = re.compile(r'([a-zA-Z0-9_]*)\.nii')
   with open(makefile, "a") as mk:
       mk.write("%s_fix_orient:\n" % id)
       for i in nii_list:
           fname = re.findall(file_name_pattern, i)
           mk.write("\tfsl5.0-fslreorient2std %s %s\n" % (i, os.path.join(outpath, "%s.nii" % fname[0])))
   print ("use \"make %s_fix_orient >> %s\" to fix orientation" % (id, os.path.join(logpath, "%s_fix_orient.log" % id)))
   return

def crop_fig(id, gopath, outpath, makefile, logpath="./"):
    nii_list = get_nii(gopath)
    file_name_pattern = re.compile(r'([a-zA-Z0-9_]*)\.nii')
    # subprocess.call("cp -r %s/* %s" % (gopath, outpath), shell=True)
    with open(makefile, "a") as mk:
        mk.write("%s_crop_fig:\n" % id)
        #mk.write("\tcp %s/* %s\n" % (gopath, outpath))
        #new_nii_list = get_nii(outpath)
        for i in nii_list:
            fname = re.findall(file_name_pattern, i)
            #mk.write("\trobustfov %s\n" % i)
            mk.write("\tfsl5.0-robustfov -i %s -r %s\n" % (i, os.path.join(outpath, "%s.nii" % fname[0])))
    print ("use \"make %s_crop_fig >> %s\" to crop figures" % (id, os.path.join(logpath, "%s_crop_fig.log" % id)))
    return


def skull_strip(id, gopath, outpath, makefile, logpath="./"):
    nii_list = get_nii(gopath)
    file_name_pattern = re.compile(r'([a-zA-Z0-9_]*)\.nii')
    with open(makefile, "a") as mk:
        mk.write("%s_skull_strip:\n" % id)
        for i in nii_list:
            fname = re.findall(file_name_pattern, i)
            mk.write("\tfsl5.0-bet %s %s\n" % (i, os.path.join(outpath, "%s.nii" % fname[0]))) # -B or -f
    print ("use \"make %s_skull_strip >> %s\" to strip skull" % (id, os.path.join(logpath, "%s_skull_strip.log" % id)))
    return

def apply_flirt(id, gopath, outpath, matrixpath, makefile, logpath="./"):
    ref_file = "/home/ubuntu/admri_code/code/utils/MNI152_T1_2mm_brain.nii.gz"
    nii_list = get_nii(gopath)
    file_name_pattern = re.compile(r'([a-zA-Z0-9_]*)\.nii')
    with open(makefile, "a") as mk:
        mk.write("%s_apply_flirt:\n" % id)
        for i in nii_list:
            fname = re.findall(file_name_pattern, i)
            mk.write("\tfsl5.0-flirt -interp spline -dof 12 -in %s -ref %s -dof 12 -omat %s -out %s\n" % (i, ref_file, os.path.join(matrixpath, "%s.mat" % fname[0]), os.path.join(outpath, "%s.nii" % fname[0]))) # -B or -f
    print ("use \"make %s_apply_flirt >> %s\" to apply flirt" % (id, os.path.join(logpath, "%s_apply_flirt.log" % id)))
    return

def apply_fnirt(id, gopath, outpath, matrixpath, makefile, logpath="./"):
    ref_file = "/usr/local/fsl/data/standard/MNI152_T1_2mm_brain"
    ref_mask_file = "MNI152_T1_2mm_brain_mask_dil1"
    conf_file = "/usr/local/fsl/etc/flirtsch/T1_2_MNI152_2mm.cnf"
    nii_list = get_nii(gopath)
    file_name_pattern = re.compile(r'([a-zA-Z0-9_]*)\.nii')
    with open(makefile, "a") as mk:
        mk.write("%s_apply_fnirt:\n" % id)
        for i in nii_list:
            fname = re.findall(file_name_pattern, i)
            mk.write("\tfnirt --in=%s --ref=%s --iout=%s --config=%s --aff=%s --refmask=%s\n" % (i, ref_file, os.path.join(outpath, "%s.nii" % fname[0]), conf_file, os.path.join(matrixpath, "%s.mat" % fname[0]), ref_mask_file))
    print ("use \"make %s_apply_fnirt >> %s\" to apply fnirt" % (id, os.path.join(logpath, "%s_apply_fnirt.log" % id)))
    return

def fix_visualization(fig_num, file_dict, filename):
    group_num = len(file_dict)
    f, ax = plt.subplots(group_num, fig_num, sharex=True, sharey=True)
    group_indx = 0
    sorted_keys = sorted(file_dict.keys())
    for i in sorted_keys:
        file_list = file_dict[i]
        sorted_file_list = sorted(file_list)
        fig_indx = 0
        pick = sorted_file_list[:fig_num]
        #target_files = [file_list[each] for each in pick]
        for each_nii in pick:
            nii_file = nib.load(each_nii)
            nii_data = nii_file.get_data()
            middle_last = int(nii_data.shape[2]/2)
            ax[group_indx, fig_indx].imshow(nii_data[:, :, middle_last], cmap='gray')
            ax[group_indx, fig_indx].set_title("%s" % i)
            fig_indx += 1
        group_indx += 1
    plt.savefig("%s.png" % filename)
    return

# i, ref_file, os.path.join(outpath, "%s.nii" % fname[0]),conf_file, os.path.join(matrixpath, "%s.mat" % fname[0] ,ref_mask_file
#
#
# /usr/local/fsl/bin/fnirt --in=T1_biascorr --ref=/usr/local/fsl/data/standard/MNI152_T1_2mm --fout=T1_to_MNI_nonlin_field --jout=T1_to_MNI_nonlin_jac --iout=T1_to_MNI_nonlin --logout=T1_to_MNI_nonlin.txt --cout=T1_to_MNI_nonlin_coeff --config=/usr/local/fsl/etc/flirtsch/T1_2_MNI152_2mm.cnf --aff=T1_to_MNI_lin.mat --refmask=MNI152_T1_2mm_brain_mask_dil1

# flirt -interp spline -dof 12 -in T1_biascorr -ref /usr/local/fsl/data/standard/MNI152_T1_2mm -dof 12 -omat T1_to_MNI_lin.mat -out T1_to_MNI_lin
