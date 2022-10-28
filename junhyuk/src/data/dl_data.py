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

def dl_data_load(args):

    ######################## DATA LOAD
    """
    users:
        dataframe
        user_id, location, age
    """
    users = pd.read_csv(args.DATA_PATH + 'users.csv')
    # users = users.drop(['location'], axis=1)
    """
    books:
        dataframe
        isbn, book_title, book_author, year_of_publication, publisher, img_url, language, category, summary, img_path
    """
    books = pd.read_csv(args.DATA_PATH + 'books.csv')
    """
    train:
        user_id, isbn, rating
    """
    train = pd.read_csv(args.DATA_PATH + 'train_ratings.csv')
    test = pd.read_csv(args.DATA_PATH + 'test_ratings.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')
    
    # if args.MODEL == 'NCF':
    # # age 전처리
    #     train = pd.merge(left=train, right=users, how='inner', on='user_id')
    #     train['age'] = train['age'].fillna(int(train['age'].mean()))
    #     # age 고유 개수들에 비해 EMBED_DIM이 크면, index 범위 넘은 에러남 -> 아닌 것 같음
    #     train['age'] = train['age'].apply(age_map)
    #     test = pd.merge(left=test, right=users, how='inner', on='user_id')
    #     test['age'] = test['age'].fillna(int(test['age'].mean()))
    #     test['age'] = test['age'].apply(age_map)
        
    #     sub = pd.merge(left=sub, right=users, how='inner', on='user_id')
    #     sub['age'] = sub['age'].fillna(int(sub['age'].mean()))
    #     sub['age'] = sub['age'].apply(age_map)

    # 모든 유저
    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    # 모든 책
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()
    # # 모든 연령
    # ages = pd.concat([train['age'], test['age']]).unique()
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
    
    # idx2age = {idx+1:age for idx, age in enumerate(ages)}

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
    
    # age2idx = {age:idx+1 for idx, age in enumerate(ages)}

    # 오 이런게 가능한가보네
    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    
    # train['age'] = train['age'].map(isbn2idx)
    # sub['age'] = sub['age'].map(isbn2idx)
    # test['age'] = test['age'].map(isbn2idx)

    if args.MODEL == 'NCF':
        """
        field_dims:
            [유저 전체 수, 아이템 전체 수, 연령대==6]
        """
        field_dims = np.array([len(user2idx), len(isbn2idx)], dtype=np.uint32)
    else:
        """
        field_dims:
            [유저 전체 수, 아이템 전체 수]
        """
        field_dims = np.array([len(user2idx), len(isbn2idx)], dtype=np.uint32)

    data = {
            # 'train': user_id, isbn, age, rating
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
