import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as spst
import sklearn
from catboost import CatBoostClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

path = "./data/"
book = pd.read_csv(path+"books.csv")
users = pd.read_csv(path+"users.csv")
train_ratings = pd.read_csv(path+"train_ratings.csv")
data = pd.merge(pd.merge(book, train_ratings, on="isbn"), users, on="user_id")

data["location_city"] = data["location"].apply(lambda x : x.split(',')[0])
data["location_state"] = data["location"].apply(lambda x : x.split(',')[1])
data["location_country"] =  data["location"].apply(lambda x : x.split(',')[2])

column = 'category'
data.loc[~data[column].isna(), column] = data.loc[~data[column].isna(), column].apply(lambda x : re.sub("[\'\[\]]","",x).lower())
data[column].describe().to_frame()

fill_nan = data[["book_title", "year_of_publication", "publisher", "language", "category", "user_id", "age", "location_city", "location_state", "location_country"]]
for c in fill_nan.columns:
    le = LabelEncoder()
    fill_nan[c] = le.fit_transform(fill_nan[c])
imp = IterativeImputer(
    estimator=[(CatBoostClassifier(verbose=False), slice(1, 10))],
    transformers=[(OneHotEncoder(sparse=False), slice(1, 10))],
    initial_strategy="most_frequent",
    verbose=1
)
X_filled = imp.fit_transform(fill_nan)
print(X_filled)