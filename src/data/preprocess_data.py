import inspect
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

def preprocess_age(users:pd.DataFrame):
    if not isinstance(users, pd.DataFrame):
        raise Exception(f"Error at {inspect.currentframe().f_code.co_name}\nnot pd.DataFrame")
    else:
        # 남은 users['age'] 결측치
        # global users['age']로 결측치 채우기
        users['age'] = users['age'].fillna(users['age'].mean())
        return users

def preprocess_location(users:pd.DataFrame):
    if not isinstance(users, pd.DataFrame):
        raise Exception(f"Error at {inspect.currentframe().f_code.co_name}\nnot pd.DataFrame")
    else:
    # location 결측치 채우기
    # 우선 location_country 결측치를 최빈 country로 채우기
        users['location_country'] = users['location_country'].fillna(users['location_country'].mode()[0])
    # state 최빈값 대치
        state_mode = users.groupby(['location_country'])['location_state'].agg(pd.Series.mode)
        idx = users[(users['location_state'].isna())].index
        for i in idx:
            try:
                tmp_country = users.loc[i, 'location_country']
                if isinstance(state_mode[tmp_country], str):
                    users.loc[i, 'location_state'] = state_mode[tmp_country]
                else:
                    users.loc[i, 'location_state'] = state_mode[tmp_country][0]
            except:
                pass
    # city 최빈값 대치
        city_mode1 = users.groupby(['location_country','location_state'])['location_city'].agg(pd.Series.mode)
        city_mode2 = users.groupby(['location_state'])['location_city'].agg(pd.Series.mode)
        city_mode3 = users.groupby(['location_country'])['location_city'].agg(pd.Series.mode)

        idx = users[(users['location_city'].isna())].index
        for i in idx:
            tmp_state = users.loc[i, 'location_state']
            tmp_country = users.loc[i, 'location_country']
            try:
                if isinstance(city_mode1[tmp_country,tmp_state], str):
                    users.loc[i, 'location_city'] = city_mode1[tmp_country, tmp_state]
                else:
                    users.loc[i, 'location_city'] = city_mode1[tmp_country, tmp_state][0]
            except:
                try:
                    if isinstance(city_mode2[tmp_state], str):
                        users.loc[i, 'location_city'] = city_mode2[tmp_state]
                    else:
                        users.loc[i, 'location_city'] = city_mode2[tmp_state][0]
                except:
                    try:
                        if isinstance(city_mode3[tmp_country], str):
                            users.loc[i, 'location_city'] = city_mode3[tmp_country]
                        else:
                            users.loc[i, 'location_city'] = city_mode3[tmp_country][0]
                    except:
                        pass
    # 너무 특이한 국가에서 사는 사람
        users['location_state'] = users['location_state'] = users['location_state'].fillna(users['location_state'].mode()[0])
        users['location_city'] = users['location_city'] = users['location_city'].fillna(users['location_city'].mode()[0])
    return users

def preprocess_publisher(books:pd.DataFrame):
    if not isinstance(books, pd.DataFrame):
        raise Exception(f"Error at {inspect.currentframe().f_code.co_name}\nnot pd.DataFrame")
    else:
    # isbn 첫 네자리 활용하여 publisher 전처리
        publisher_dict=(books['publisher'].value_counts()).to_dict()
        publisher_count_df= pd.DataFrame(list(publisher_dict.items()),columns = ['publisher','count'])
        publisher_count_df = publisher_count_df.sort_values(by=['count'], ascending = False)

        modify_list = publisher_count_df[publisher_count_df['count']>1].publisher.values
        for publisher in modify_list:
            try:
                number = books[books['publisher']==publisher]['isbn'].apply(lambda x: x[:4]).value_counts().index[0]
                right_publisher = books[books['isbn'].apply(lambda x: x[:4])==number]['publisher'].value_counts().index[0]
                books.loc[books[books['isbn'].apply(lambda x: x[:4])==number].index,'publisher'] = right_publisher
            except: 
                pass
        return books

def users2idx(context_df, train_df, test_df, feature2idx:dict):
    def age_map(x: int) -> int:
        x = int(x)
        if x < 20:
            return 1
        elif x >= 20 and x < 30:
            return 2
        elif x >= 30 and x < 40:
            return 3
        elif x >= 40 and x < 50:
            return 4
        elif x >= 50 and x < 60:
            return 5
        else:
            return 6
    if not isinstance(context_df, pd.DataFrame) or not isinstance(train_df, pd.DataFrame) or \
        not isinstance(test_df, pd.DataFrame):
        raise Exception(f"Error at {inspect.currentframe().f_code.co_name}\nnot pd.DataFrame")
    else:
        train_df['age'] = train_df['age'].fillna(int(train_df['age'].mean()))
        train_df['age'] = train_df['age'].apply(age_map)
        test_df['age'] = test_df['age'].fillna(int(test_df['age'].mean()))
        test_df['age'] = test_df['age'].apply(age_map)
        
        for feature_name in ['city', 'state', 'country']:
            idx_name = 'loc' + feature_name + '2idx'
            feature_name = 'location_' + feature_name
            
            feature2idx[idx_name] = {v:k for k,v in enumerate(context_df[feature_name].unique())}
            train_df[feature_name] = train_df[feature_name].map(feature2idx[idx_name])
            test_df[feature_name] = test_df[feature_name].map(feature2idx[idx_name])
        
        return feature2idx, train_df, test_df

def books2idx(context_df, train_df, test_df, features_name:list, feature2idx:dict):
    if not isinstance(context_df, pd.DataFrame) or not isinstance(train_df, pd.DataFrame) or \
        not isinstance(test_df, pd.DataFrame):
        raise Exception(f"Error at {inspect.currentframe().f_code.co_name}\nnot pd.DataFrame")
    else:
        for feature_name in features_name:
            if feature_name == 'isbn' or feature_name == 'user_id':
                continue
            idx_name = feature_name + '2idx'
            feature2idx[idx_name] = {v:k for k,v in enumerate(context_df[feature_name].unique())}
            train_df[feature_name] = train_df[feature_name].map(feature2idx[idx_name])
            test_df[feature_name] = test_df[feature_name].map(feature2idx[idx_name])
        return feature2idx, train_df, test_df