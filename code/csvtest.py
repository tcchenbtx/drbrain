import csv
import os

# path
base_path = os.path.abspath(os.path.dirname(__file__))
base_path = os.path.join(base_path, "..")
raw_data_path = os.path.join(base_path, "raw_data")
gwas_data_path = os.path.join(raw_data_path, "gwas")
gwas_data = os.path.join(gwas_data_path, "adni_gwas_v2_set1", "002_S_0295.csv")

with open(gwas_data, 'rb') as csvfile:
    myreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    i = 0
    for row in myreader:
        print (row)
        print ', '.join(row)
        i += 1
        if i >= 5:
            break


