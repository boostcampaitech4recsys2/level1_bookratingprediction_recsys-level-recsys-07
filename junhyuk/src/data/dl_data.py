import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

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
    
def process_context_data(users, books, ratings1, ratings2):
# 남은 users['age'] 결측치 채우기
# global users['age']로 결측치 채우기
    users['age'] = users['age'].fillna(users['age'].mean())
    
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




















def dl_data_load(args):
    ######################## DATA LOAD
    """
    users_preprocessed:
        dataframe
        user_id,age,location_city,location_state,location_country
    """
    users = pd.read_csv(args.DATA_PATH + 'users_preprocessed.csv')
    """
    books_merged:
        dataframe
        isbn,book_title,year_of_publication,publisher,img_url,
        language,summary,img_path,category_high,book_author,category,
        new_language,remove_country_code,book_author_over3,book_author_over5,
        book_author_over10,book_author_over50,book_author_over100
    """
    books = pd.read_csv(args.DATA_PATH + 'books_merged.csv')
    """
    train:
        user_id, isbn, rating
    """
    train = pd.read_csv(args.DATA_PATH + 'train_ratings.csv')
    test = pd.read_csv(args.DATA_PATH + 'test_ratings.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')

    # 모든 유저
    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    # 모든 책
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()
    """
    dict 생성
        key: idx
        value: user_id
    """
    idx2user = {idx:id for idx, id in enumerate(ids)}
    """
    dict 생성
        key: idx
        value: item_id
    """
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}
    """
    dict 생성
        key: user_id
        value: idx
    """
    user2idx = {id:idx for idx, id in idx2user.items()}
    """
    dict 생성
        key: item_id
        value: idx
    """
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}
    
    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    
    idx, context_train, context_test = process_context_data(users, books, train, test)
    field_dims = np.array([len(user2idx), len(isbn2idx)], dtype=np.uint32)

    data = {
            'train':train,
            'test':test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }


    return data

def dl_data_split(args, data):
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.TEST_SIZE,
                                                        random_state=args.SEED,
                                                        shuffle=True
                                                        )
    # X_train: user_id, isbn, age
    # X_valid: ratings
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data

def dl_data_loader(args, data):
    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE, drop_last=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

    # 'train_dataloader':
        # X_train: user_id, isbn, age
        # X_valid: ratings
    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
