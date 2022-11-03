import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

from preprocess_data import *

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
    
def process_context_data(users, books, ratings1, ratings2, features_name:list):
    # publisher 전처리
    # books = preprocess_publisher(books)
    # age, location 전처리
    users = preprocess_location(preprocess_age(users))
    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)
    # users = users.dropna()
    # books = books.dropna()

    # print(type(features_name))
    
    # 인덱싱 처리된 데이터 조인
    """
    users_preprocessed:
        dataframe
        user_id,age,location_city,location_state,location_country
        
    books_merged:
        dataframe
        isbn,book_title,year_of_publication,publisher,img_url,
        language,summary,img_path,category_high,book_author,category,
        new_language,remove_country_code,book_author_over3,book_author_over5,
        book_author_over10,book_author_over50,book_author_over100
    """
    # breakpoint()
    # user_id, isbn, age, city, state, country, category_high, publisher_4_digit, language, author_10
    # context_df = ratings.merge(users, on='user_id', how='left').merge(books[features_name], on='isbn', how='left')
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[features_name], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[features_name], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[features_name], on='isbn', how='left')

    idx = dict()
    # users 인덱싱
    # 3개 추가
    idx, train_df, test_df = users2idx(context_df, train_df, test_df, idx)
    
    # books 인덱싱
    idx, train_df, test_df = books2idx(context_df, train_df, test_df, features_name, idx)
    
    return idx, train_df, test_df

def dl_data_load(args):
    # user_id, isbn, age, city, state, country, category_high, publisher_4_digit, language, author_10
    features_name = args.ADD_CONTEXT
    
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
    train = pd.read_csv(args.DATA_PATH + 'train_ratings.csv')
    test = pd.read_csv(args.DATA_PATH + 'test_ratings.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')

    # 모든 유저
    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    # 모든 책
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()
    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}
    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}
    
    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)
    users['user_id'] = users['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    books['isbn'] = books['isbn'].map(isbn2idx)
    
    idx, context_train, context_test = process_context_data(users, books, train, test, features_name)
    field_dims = np.array([len(user2idx), len(isbn2idx), 6], dtype=np.uint32)
    for idx_name in idx.keys():
        field_dims = np.append(field_dims, len(idx[idx_name]))
        
    field_name_dim_dict = {'user_id':np.array((0, ), dtype=np.long),
                           'isbn':np.array((1, ), dtype=np.long),
                           'age': np.array((2, ), dtype=np.long)}
    for i, idx_name in enumerate(idx.keys()):
        field_name_dim_dict[idx_name] = np.array((i + 3, ), dtype=np.long)

    data = {
            'train':context_train,
            'test':context_test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            'field_name_dim_dict':field_name_dim_dict
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
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    """
    data['X_train']:
        user_id, isbn, age, city, state, country, feature_names....
    """
    return data

def dl_data_loader(args, data):
    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE, drop_last=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
