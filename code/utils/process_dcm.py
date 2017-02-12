import os
import re
import collections
# # path
# base_path = os.path.abspath(os.path.dirname(__file__))
# base_path = os.path.join(base_path, "..", "..")
# code_path = os.path.join()
# raw_data_path = os.path.join(base_path, "raw_data")
# log_path = os.path.join(code_path, "log")


# convert dcm to nii files
def batch_dcm2nii(id, gopath, makefile_path, log_path):
    pathinfo = []
    datamix_check = []
    mix_found = []
    for root, dirs, files in os.walk(gopath):
        for file in files:
            if re.match(r'.*\.dcm$', file):
                pathinfo.append(root)
            if re.match(r'.*\.xml$', file):
                datamix_check.append(root)

    counter = collections.Counter(datamix_check)
    download_info = counter.most_common(1)

    print ("downloaded file numbers:")
    print ("in %s, %d files are downloaded" % (download_info[0][0], download_info[0][1]))
    print ("-"*80)
    print ("potential mixed dataset:")
    for i, j in counter.items():
        if j >= 2 and j < download_info[0][1]:
            print ("%s : %d data are mixed" % (i, j))
            mix_found.append(i)

    to_convert = sorted(list(set(pathinfo)))
    final_to_convert = [i for i in to_convert if i not in mix_found]

    print("-"*80)
    print("After removing data with mixed dataset:")
    print("there are %d files for dcm2nii convertion\n" % len(final_to_convert))

    # construct makefile for dcm2nii command
    dbid_pattern = re.compile(r'%s/([A-Z]*)/' % gopath)
    subid_pattern = re.compile(r'%s/[A-Z]*/(\w*)/' % gopath)
    imgid_pattern = re.compile(r'/(\w*)$')

    subject_list_file = os.path.join(gopath, "%s_subject.txt" % id)

    with open(makefile_path, 'a') as mk, open(subject_list_file, 'a') as subfile:
        mk.write("%s_batch_dcm2nii:\n" % id)
        for i in final_to_convert:
            i_subid = re.findall(subid_pattern, i)
            i_imgid = re.findall(imgid_pattern, i)
            i_dbid = re.findall(dbid_pattern, i)
            mk.write("\tdcm2niix -o %s -f %s_%s_%s %s\n" % (root, i_dbid[0], i_subid[0], i_imgid[0], i))
            subfile.write("%s,%s,%s\n" % (i_dbid[0], i_subid[0], i_imgid[0]))

    print('Use "make %s_batch_dcm2nii >> %s" to generate nii files for %s group' % (id, os.path.join(log_path, "%s_batch_dcm2nii.log" % id), id))
    print("subject info are saved in %s" % subject_list_file)

    return


# deal with the naming issue for files
def batch_move_rename(id, gopath, makefile_path, log_path):
    nii_pattern = re.compile(r'.*\.nii$|.*\.nii.gz$')
    dbid_pattern = re.compile(r'%s/([A-Z]*)/' % gopath)
    subid_pattern = re.compile(r'%s/[A-Z]*/(\w*)/' % gopath)
    imgid_pattern = re.compile(r'/(\w*)$')

    subject_list_file = os.path.join(gopath, "%s_subject.txt" % id)
    with open(makefile_path, 'a') as mk, open(subject_list_file, 'a') as subfile:
        mk.write("%s_batch_move_rename:\n" % id)
        count = 0
        for root, dirs, files in os.walk(gopath):
            for each_file in files:
                if re.match(nii_pattern, each_file):
                    i_subid = re.findall(subid_pattern, root)
                    i_imgid = re.findall(imgid_pattern, root)
                    i_dbid = re.findall(dbid_pattern, root)
                    mk.write("\tcp %s %s\n" % (os.path.join(root, each_file), os.path.join(gopath, "%s_%s_%s.nii" % (i_dbid[0], i_subid[0], i_imgid[0]))))
                    subfile.write("%s,%s,%s\n" % (i_dbid[0], i_subid[0], i_imgid[0]))
                    count += 1
    print ("there are %d data for %s group" % (count, id))

    print('Use "make %s_batch_move_rename >> %s" to generate move and rename files for %s group' % (id, os.path.join(log_path, "%s_batch_move_rename.log" % id), id))
    print("subject info are saved in %s" % subject_list_file)

    return