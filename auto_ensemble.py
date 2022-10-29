import os
from itertools import combinations as comb
PATH = "/opt/ml/input/code/level1_bookratingprediction_recsys-level-recsys-07/submit/"
file_list = os.listdir(PATH)

MODELS = ['FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN']
REQUIRE_MODELS = ["CNN_FM"]
essencial = list(map(lambda x:f"_{x}.csv",REQUIRE_MODELS))
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
def get_ensemble_cmd(model_name, strategy="WEIGHTED", weight_list = []): 
    weight = ' '.join(list(map(str, weight_list)))
    if strategy == "MIXED": 
        return f"python ensemble.py --ENSEMBLE_FILES {model_name} --ENSEMBLE_STRATEGY MIXED"
    if weight_list:
        return f"python ensemble.py --ENSEMBLE_FILES {model_name} --ENSEMBLE_STRATEGY WEIGHTED --ENSEMBLE_WEIGHT {weight}"
    return f"python ensemble.py --ENSEMBLE_FILES {model_name} --ENSEMBLE_STRATEGY WEIGHTED"

for i in get_ensemble_list(optional_files, essencial_files):
    print(get_ensemble_cmd(i))
    os.system(get_ensemble_cmd(i))
