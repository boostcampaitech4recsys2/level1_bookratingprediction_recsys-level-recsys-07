import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from ._models import _NeuralCollaborativeFiltering, _WideAndDeepModel, _DeepCrossNetworkModel
from ._models import rmse, RMSELoss

import time

class NeuralCollaborativeFiltering:

    """
    data 딕셔너리의 구성
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
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid']
    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader']
    """
    def __init__(self, args, data):
        super().__init__()

        # torch.sqrt(nn.MSELoss(pred, target)+self.eps)
        self.criterion = RMSELoss()

        """
        X:
            batch_size * 2 == batch_size * len([user_id_열, isbn_열])
        y:
            batch_size * 1 == batch_size * len([ratings_열])
        """
        self.train_dataloader = data['train_dataloader']
        """
        X:
            batch_size * 2 == batch_size * len([user_id_열, isbn_열])
        y:
            batch_size * 1 == batch_size * len([ratings_열])
        """
        self.valid_dataloader = data['valid_dataloader']
        """
        field_dims:
            [유저 전체 수, 아이템 전체 수]
        """
        self.field_dims = data['field_dims']
        # self.train_dataloader에서 0번째 열이 user_id라는 뜻
        self.user_field_idx = np.array((0, ), dtype=np.long)
        # self.train_dataloader에서 1번째 열이 isbn이라는 뜻
        self.item_field_idx = np.array((1, ), dtype=np.long)
        
        # # self.train_dataloader에서 2번째 열이 age이라는 뜻
        # self.age_field_idx = np.array((2, ), dtype=np.long)

        
        self.embed_dim = args.NCF_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = args.DEVICE

        self.mlp_dims = args.NCF_MLP_DIMS
        self.dropout = args.NCF_DROPOUT

        self.model = _NeuralCollaborativeFiltering(self.field_dims, user_field_idx=self.user_field_idx, item_field_idx=self.item_field_idx,
                                                    embed_dim=self.embed_dim, mlp_dims=self.mlp_dims, dropout=self.dropout).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)


    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        sum_time = 0
        train_loss_list = []
        valid_loss_list = []
        train_acc_list = []
        valid_acc_list = []
        print_iter = 1

        for epoch in range(self.epochs):
            start_time = time.time()
            train_pred = []
            train_true = []
            self.model.train()
            train_loss = 0
            tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                train_true.append(target.cpu().numpy())
                train_pred.append(y.detach().cpu().numpy())
                
            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)
            
            # train 정보
            train_loss /= (i + 1)
            train_loss_list.append(train_loss)
            train_rmse_score = rmse(train_true, train_pred)
            train_acc_list.append(train_rmse_score)
            
            # valid 정보
            valid_rmse_score, valid_loss = self.predict_train()
            valid_loss_list.append(valid_loss)
            valid_acc_list.append(valid_rmse_score)
            
            now_time = time.time()
            elapsed = now_time - start_time
            
            if epoch % print_iter == 0:
                print(f"Epoch: {epoch}, elapsed: {elapsed}")
                print(f"train acc: {train_rmse_score}, train loss: {train_loss}")
                print(f"valid acc: {valid_rmse_score}, valid loss: {valid_loss}")
            sum_time += elapsed
            
        print(f"train done! elapsed: {sum_time}")
        
        log_df = pd.DataFrame({'train loss': train_loss_list,\
            'train acc': train_acc_list,\
            'val loss': valid_loss_list,\
            'val acc': valid_acc_list})
        log_df.to_csv("NCF log.csv", mode="w")

    def predict_train(self):
        self.model.eval()
        valid_loss = 0
        targets, predicts = list(), list()
        with torch.no_grad():
            for i, (fields, target) in enumerate(tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0)):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                valid_loss += loss.item()
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
            valid_loss /= (i + 1)
        return rmse(targets, predicts), valid_loss

    def predict(self, dataloader):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts


class WideAndDeepModel:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()

        self.train_dataloader = data['train_dataloader']
        self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']

        self.embed_dim = args.WDN_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = args.DEVICE

        self.mlp_dims = args.WDN_MLP_DIMS
        self.dropout = args.WDN_DROPOUT

        self.model = _WideAndDeepModel(self.field_dims, self.embed_dim, mlp_dims=self.mlp_dims, dropout=self.dropout).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)


    def train(self):
        # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        sum_time = 0
        train_loss_list = []
        valid_loss_list = []
        train_acc_list = []
        valid_acc_list = []
        print_iter = 1

        for epoch in range(self.epochs):
            start_time = time.time()
            train_pred = []
            train_true = []
            self.model.train()
            train_loss = 0
            tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                train_true.append(target.cpu().numpy())
                train_pred.append(y.detach().cpu().numpy())
                
            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)
            
            # train 정보
            train_loss /= (i + 1)
            train_loss_list.append(train_loss)
            train_rmse_score = rmse(train_true, train_pred)
            train_acc_list.append(train_rmse_score)
            
            # valid 정보
            valid_rmse_score, valid_loss = self.predict_train()
            valid_loss_list.append(valid_loss)
            valid_acc_list.append(valid_rmse_score)
            
            now_time = time.time()
            elapsed = now_time - start_time
            
            if epoch % print_iter == 0:
                print(f"Epoch: {epoch}, elapsed: {elapsed}")
                print(f"train acc: {train_rmse_score}, train loss: {train_loss}")
                print(f"valid acc: {valid_rmse_score}, valid loss: {valid_loss}")
            sum_time += elapsed
            
        print(f"train done! elapsed: {sum_time}")
        
        log_df = pd.DataFrame({'train loss': train_loss_list,\
            'train acc': train_acc_list,\
            'val loss': valid_loss_list,\
            'val acc': valid_acc_list})
        log_df.to_csv("WDN log.csv", mode="w")
    
    
    def predict_train(self):
        self.model.eval()
        valid_loss = 0
        targets, predicts = list(), list()
        with torch.no_grad():
            for i, (fields, target) in enumerate(tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0)):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                valid_loss += loss.item()
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
            valid_loss /= (i + 1)
        return rmse(targets, predicts), valid_loss


    def predict(self, dataloader):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts


class DeepCrossNetworkModel:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()

        self.train_dataloader = data['train_dataloader']
        self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']

        self.embed_dim = args.DCN_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = args.DEVICE

        self.mlp_dims = args.DCN_MLP_DIMS
        self.dropout = args.DCN_DROPOUT
        self.num_layers = args.DCN_NUM_LAYERS

        self.model = _DeepCrossNetworkModel(self.field_dims, self.embed_dim, num_layers=self.num_layers, mlp_dims=self.mlp_dims, dropout=self.dropout).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)


    def train(self):
        # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        sum_time = 0
        train_loss_list = []
        valid_loss_list = []
        train_acc_list = []
        valid_acc_list = []
        print_iter = 1

        for epoch in range(self.epochs):
            start_time = time.time()
            train_pred = []
            train_true = []
            self.model.train()
            train_loss = 0
            tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                train_true.append(target.cpu().numpy())
                train_pred.append(y.detach().cpu().numpy())
                
            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)
            
            # train 정보
            train_loss /= (i + 1)
            train_loss_list.append(train_loss)
            train_rmse_score = rmse(train_true, train_pred)
            train_acc_list.append(train_rmse_score)
            
            # valid 정보
            valid_rmse_score, valid_loss = self.predict_train()
            valid_loss_list.append(valid_loss)
            valid_acc_list.append(valid_rmse_score)
            
            now_time = time.time()
            elapsed = now_time - start_time
            
            if epoch % print_iter == 0:
                print(f"Epoch: {epoch}, elapsed: {elapsed}")
                print(f"train acc: {train_rmse_score}, train loss: {train_loss}")
                print(f"valid acc: {valid_rmse_score}, valid loss: {valid_loss}")
            sum_time += elapsed
            
        print(f"train done! elapsed: {sum_time}")
        
        log_df = pd.DataFrame({'train loss': train_loss_list,\
            'train acc': train_acc_list,\
            'val loss': valid_loss_list,\
            'val acc': valid_acc_list})
        log_df.to_csv("WDN log.csv", mode="w")
    
    
    def predict_train(self):
        self.model.eval()
        valid_loss = 0
        targets, predicts = list(), list()
        with torch.no_grad():
            for i, (fields, target) in enumerate(tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0)):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                valid_loss += loss.item()
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
            valid_loss /= (i + 1)
        return rmse(targets, predicts), valid_loss


    def predict(self, dataloader):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts
