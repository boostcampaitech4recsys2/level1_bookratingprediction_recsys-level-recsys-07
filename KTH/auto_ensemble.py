import os
from itertools import combinations as comb
os.system("cd ..")
PATH = "/opt/ml/input/code/submit"
file_list = os.listdir(PATH)

MODELS = ['FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN']
essencial =["_NCF.csv"]
ENSEMBLE_STRATEGY = "MIXED"


def split_essencial_file(files, essencial_list):
    optional = []
    essencial = []
    for _file in files:
        if not _file.endswith(".csv"):
            continue
        if "-" in _file[16:]:
            continue
        if True in [_file.endswith(x) for x in essencial_list]:
            essencial.append(_file.rstrip(".csv"))
        else:
            optional.append(_file.rstrip(".csv"))
    return optional, essencial

optional_files, essencial_files = split_essencial_file(file_list, essencial)

def get_ensemble_list(optional_files, essencial_files):
    enb_list = []
    offset = ','.join(essencial_files)+","
    for size in range(1, len(optional_files)+1):
        enb_list.extend(comb(optional_files, size))
    return list(map(lambda x: offset+','.join(x),enb_list))

# strategy -> WEIGHTED, MIXED
def get_ensemble_cmd(enb_file_name): 
    return f"python ensemble.py --ENSEMBLE_FILES {enb_file_name} --ENSEMBLE_STRATEGY {ENSEMBLE_STRATEGY}"

for i in get_ensemble_list(optional_files, essencial_files):
    print(get_ensemble_cmd(i))
    os.system(get_ensemble_cmd(i))
