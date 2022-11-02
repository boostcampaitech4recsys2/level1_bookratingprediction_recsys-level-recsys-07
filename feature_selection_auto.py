import os
import pandas as pd
from itertools import combinations

#### Settings ####
BOOK_PATH = "/opt/ml/input/code/level1_bookratingprediction_recsys-level-recsys-07/data/books_data/books.csv"
MODEL = "NCF"
MIN_LEN = 10

#### File Settings ####
CONTAIN_EMBEDDINGS = False
ESSENCIAL = ["book_author_over100","year_of_publication"]
IGNORE = ["isbn","book_title","remove_country_code","img_path"]
if not CONTAIN_EMBEDDINGS:
    IGNORE = IGNORE + [str(i) for i in range(512)]


#### Parameters ####
## NCF ##
BATCH_SIZE = 1024
EPOCH = 10
LR = 1e-3
WEIGHT_DECAY = 1e-6
NCF_EMBED_DIM = 16
NCF_MLP_DIMS = ' '.join(list(map(str,[256,256,256,256,256])))
DATA_PATH = "data/"
"""
 "book_title",
 "year_of_publication",
 "publisher",
 "img_url",
 "language",
 "summary",
 "img_path",
 "category_high",
 "book_author",
 "category",
 "new_language",
 "remove_country_code",
 "book_author_over3",
 "book_author_over5",
 "book_author_over10",
 "book_author_over50",
 "book_author_over100",
"""


available = [i for i in pd.read_csv(BOOK_PATH).columns if i not in IGNORE and not i.startswith("Unnamed")]

fix_list = list(filter(lambda x: x in ESSENCIAL,available))
iter_list = list(filter(lambda x: x not in ESSENCIAL,available))

for i in range(MIN_LEN-len(fix_list), len(iter_list)+1):
    for case in combinations(iter_list, i):
        cmd = {
          "model": f"--MODEL {MODEL}",
          "features":f"--ADD_CONTEXT {' '.join(list(case)+fix_list)}",
          "batch_size": f"--BATCH_SIZE {BATCH_SIZE}",
          "epoch": f"--EPOCH {EPOCH}",
          "lr" : f"--LR {LR}",
          "weight_decay": f"--WEIGHT_DECAY {WEIGHT_DECAY}",
          "NCF_EMBED_DIM": f"--NCF_EMBED_DIM {NCF_EMBED_DIM}",
          "NCF_MLP_DIMS": f"--NCF_MLP_DIMS {NCF_MLP_DIMS}",
          "DATA_PATH": f"--DATA_PATH {DATA_PATH}"
        }
        print("python main.py "+' '.join(cmd.values()))
        os.system("python main.py "+' '.join(cmd.values()))
