import os
from utils import process_nii
from utils import img_check

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
MCI_afterflirt_path = os.path.join(preprocess_data_path, "after_flirt", "MCI")
Normal_afterflirt_path = os.path.join(preprocess_data_path, "after_flirt", "Normal")
PD_afterflirt_path = os.path.join(preprocess_data_path, "after_flirt", "PD")

# get nii paths for all data

AD_nii_files = process_nii.get_nii(AD_afterflirt_path)
MCI_nii_files = process_nii.get_nii(MCI_afterflirt_path)
Normal_nii_files = process_nii.get_nii(Normal_afterflirt_path)
PD_nii_files = process_nii.get_nii(PD_afterflirt_path)

Normal_nii_first = Normal_nii_files[:737]
Normal_nii_second = Normal_nii_files[737:]


#img_check.check_all(AD_nii_files, "Global_AD_3")
#img_check.check_all(MCI_nii_files, "Global_MCI")
#img_check.check_all(Normal_nii_first, "Global_Normal_1")
#img_check.check_all(Normal_nii_second, "Global_Normal_2")
#img_check.check_all(PD_nii_files, "Global_PD")

total_bad = []
print ("for AD:")
AD_bad = [277, 297, 320, 374, 386, 498, 526, 729, 730, 734]
for i in AD_bad:
    i += 1
    total_bad.append(AD_nii_files[i])
    print(AD_nii_files[i])

MCI_bad = [154, 251, 396, 432, 452, 607, 691]
for i in MCI_bad:
    i += 1
    total_bad.append(MCI_nii_files[i])
    print(MCI_nii_files[i])

Normal_bad = [308, 337, 350, 390, 625, 765, 775, 849, 985, 1166, 1198, 1386]
for i in Normal_bad:
    i += 1
    total_bad.append(Normal_nii_files[i])
    print(Normal_nii_files[i])

PD_bad = [39, 44, 61, 74, 97, 196, 204, 266, 320, 332, 341, 363, 428, 435, 461, 488, 504, 526, 549, 562, 564, 570, 660]
for i in PD_bad:
    i += 1
    total_bad.append(PD_nii_files[i])
    print(PD_nii_files[i])

bad_folder = os.path.join(preprocess_data_path, "after_flirt", "Bad")


#print("total:")
#print(total_bad)

with open(make_file_path, "a") as mk:
    mk.write("move_bad:\n")
    for i in total_bad:
        mk.write("\tmv %s %s\n" % (i, bad_folder))

img_check.check_all(total_bad, "bad_files_final")








