import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

from src.data.preprocess_data import *

def age_map(x: int) -> int:
    x = int(x)
    if x < 20:
        return 0
    elif x >= 20 and x < 30:
        return 1
    elif x >= 30 and x < 40:
        return 2
    elif x >= 40 and x < 50:
        return 3
    elif x >= 50 and x < 60:
        return 4
    else:
        return 5

def process_context_data(users, books, ratings1, ratings2, features_name):
    
    # # isbn 첫 네자리 활용하여 publisher 전처리
    # publisher_dict=(books['publisher'].value_counts()).to_dict()
    # publisher_count_df= pd.DataFrame(list(publisher_dict.items()),columns = ['publisher','count'])
    # publisher_count_df = publisher_count_df.sort_values(by=['count'], ascending = False)

    # modify_list = publisher_count_df[publisher_count_df['count']>1].publisher.values
    # for publisher in modify_list:
    #     try:
    #         number = books[books['publisher']==publisher]['isbn'].apply(lambda x: x[:4]).value_counts().index[0]
    #         right_publisher = books[books['isbn'].apply(lambda x: x[:4])==number]['publisher'].value_counts().index[0]
    #         books.loc[books[books['isbn'].apply(lambda x: x[:4])==number].index,'publisher'] = right_publisher
    #     except: 
    #         pass
    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # user_power
    tmp = ratings1.groupby('user_id')['isbn'].count().reset_index()
    tmp.columns = list(tmp.columns)[:-1] + ['user_power']
    ratings = ratings.merge(tmp, how='left', on='user_id')
    ratings['user_power'] = ratings['user_power'].fillna(0)
    ratings1 = ratings1.merge(tmp, how='left', on='user_id')
    ratings2 = ratings2.merge(tmp, how='left', on='user_id')
    ratings2['user_power'] = ratings2['user_power'].fillna(0)

    # book_popularity
    tmp = ratings1.groupby('isbn')['user_id'].count().reset_index()
    tmp.columns = list(tmp.columns)[:-1] + ['book_popularity']
    ratings = ratings.merge(tmp, how='left', on='isbn')
    ratings['book_popularity'] = ratings['book_popularity'].fillna(0)
    ratings1 = ratings1.merge(tmp, how='left', on='isbn')
    ratings2 = ratings2.merge(tmp, how='left', on='isbn')
    ratings2['book_popularity'] = ratings2['book_popularity'].fillna(0)    
                    
    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[features_name], on='isbn', how='left')
    # .merge(books.filter(regex=r'(isbn)|(^\d+$)'), on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[features_name], on='isbn', how='left')
    # .merge(books.filter(regex=r'(isbn)|(^\d+$)'), on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[features_name], on='isbn', how='left')
    # .merge(books.filter(regex=r'(isbn)|(^\d+$)'), on='isbn', how='left')

    idx = dict()
    # users 인덱싱
    # 3개 추가
    idx, train_df, test_df = feat2idx(context_df, train_df, test_df, idx, use_embedding=False)
    # idx, train_df, test_df = users2idx(context_df, train_df, test_df, idx)
    
    # books 인덱싱
    # idx, train_df, test_df = books2idx(context_df, train_df, test_df, features_name, idx)
    
    # # 인덱싱 처리
    # loc_city2idx = {v:k for k,v in enumerate(context_df['location_city'].unique())}
    # loc_state2idx = {v:k for k,v in enumerate(context_df['location_state'].unique())}
    # loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}

    # train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    # train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    # train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    # test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    # test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    # test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    # train_df['age'] = train_df['age'].fillna(int(train_df['age'].mean()))
    # train_df['age'] = train_df['age'].apply(age_map)
    # test_df['age'] = test_df['age'].fillna(int(test_df['age'].mean()))
    # test_df['age'] = test_df['age'].apply(age_map)

    # # book 파트 인덱싱
    # category2idx = {v:k for k,v in enumerate(context_df['isfiction'].unique())}
    # publisher2idx = {v:k for k,v in enumerate(context_df['publisher_small'].unique())}
    # language2idx = {v:k for k,v in enumerate(context_df['isenglish'].unique())}
    # author2idx = {v:k for k,v in enumerate(context_df['book_author_over100'].unique())}
    # yearofpublication2idx = {v:k for k,v in enumerate(context_df['new_year'].unique())}
    # user_power2idx = {v:k for k,v in enumerate(context_df['user_power'].unique())}
    # book_popularity2idx = {v:k for k,v in enumerate(context_df['book_popularity'].unique())}

    # train_df['isfiction'] = train_df['isfiction'].map(category2idx)
    # train_df['publisher_small'] = train_df['publisher_small'].map(publisher2idx)
    # train_df['isenglish'] = train_df['isenglish'].map(language2idx)
    # train_df['book_author_over100'] = train_df['book_author_over100'].map(author2idx)
    # train_df['new_year'] = train_df['new_year'].map(yearofpublication2idx)
    # train_df['user_power'] = train_df['user_power'].map(user_power2idx)
    # train_df['book_popularity'] = train_df['book_popularity'].map(book_popularity2idx)

    # test_df['isfiction'] = test_df['isfiction'].map(category2idx)
    # test_df['publisher_small'] = test_df['publisher_small'].map(publisher2idx)
    # test_df['isenglish'] = test_df['isenglish'].map(language2idx)
    # test_df['book_author_over100'] = test_df['book_author_over100'].map(author2idx)
    # test_df['new_year'] = test_df['new_year'].map(yearofpublication2idx)
    # test_df['user_power'] = test_df['user_power'].map(user_power2idx)
    # test_df['book_popularity'] = test_df['book_popularity'].map(book_popularity2idx)

    # idx = {
    #     "loc_city2idx":loc_city2idx,
    #     "loc_state2idx":loc_state2idx,
    #     "loc_country2idx":loc_country2idx,
    #     "category2idx":category2idx,
    #     "publisher2idx":publisher2idx,
    #     "language2idx":language2idx,
    #     "author2idx":author2idx,
    #     "yearofpublication2idx":yearofpublication2idx,
    #     "user_power2idx":user_power2idx,
    #     "book_popularity2idx":book_popularity2idx,
    # }
    return idx, train_df, test_df


def context_data_load(args):

    ######################## DATA LOAD
    # users = pd.read_csv(args.DATA_PATH + 'users.csv')
    users = pd.read_csv(args.DATA_PATH + 'users_preprocessed.csv')
    books = pd.read_csv(args.DATA_PATH + 'books_final.csv')
    train = pd.read_csv(args.DATA_PATH + 'train_ratings.csv')
    test = pd.read_csv(args.DATA_PATH + 'test_ratings.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
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

    idx, context_train, context_test = process_context_data(users, books, train, test, args.ADD_CONTEXT)
    n_features = context_train.shape[1]-1
    field_dims = np.array([len(user2idx), len(isbn2idx)], dtype=np.uint32)
    for idx_name in idx.keys():
        field_dims = np.append(field_dims, len(idx[idx_name]))
    field_idx = np.arange(len(field_dims))
    # drop_features = [
    #             # users
    #                 #  'user_id', 
    #                 #  'age', 
    #                 #  'location_city', 
    #                 #  'location_state', 
    #                 #  'location_country', 
    #             # books
    #                 #  'isbn',
    #                 'year_of_publication',
    #                 'publisher',
    #                 'language',
    #                 'img_path',
    #                 'category_high',
    #                 'book_author',
    #                 'book_author_over3',
    #                 'book_author_over5',
    #                 'book_author_over10',
    #                 'book_author_over50',
    #                 # 'book_author_over100',
    #                 'category',
    #                 'new_language',
    #                 'remove_country_code',
    #                 # 'isfiction',
    #                 # 'isenglish',
    #                 # 'new_year',
    #                 # 'publisher_small',
    #                 # 'user_power',
    #                 # 'book_popularity',
    #                 ]
    
    # field_dims = np.array([
    #                         len(user2idx), len(isbn2idx),
    #                         6, 
    #                         # len(idx['loc_city2idx']), 
    #                         # len(idx['loc_state2idx']), 
    #                         len(idx['loc_country2idx']),
    #                         len(idx['author2idx']),
    #                         len(idx['category2idx']), 
    #                         len(idx['language2idx']),
    #                         len(idx['yearofpublication2idx']),
    #                         len(idx['publisher2idx']),
    #                         len(idx['user_power2idx']),
    #                         len(idx['book_popularity2idx']),
    #                         ], dtype=np.uint32)
    
  

    data = {
            'train':context_train,
            'test':context_test.drop(['rating'],axis=1),
            'n_features':n_features,
            'field_idx':field_idx,
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


def context_data_split(args, data):
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.TEST_SIZE,
                                                        random_state=args.SEED,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data

def context_data_loader(args, data):
    train_dataset = TensorDataset(torch.FloatTensor(data['X_train'].values), torch.FloatTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.FloatTensor(data['X_valid'].values), torch.FloatTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.FloatTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader
    return data
