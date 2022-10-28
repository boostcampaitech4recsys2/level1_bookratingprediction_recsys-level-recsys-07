import tqdm
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from ._models import _FactorizationMachineModel, _FieldAwareFactorizationMachineModel
from ._models import rmse, RMSELoss


class FactorizationMachineModel:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()

        self.train_dataloader = data['train_dataloader']
        self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']

        self.embed_dim = args.FM_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = args.DEVICE

        self.model = _FactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
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
        log_df.to_csv("FM log.csv", mode="w")
    
    
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


class FieldAwareFactorizationMachineModel:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()

        self.train_dataloader = data['train_dataloader']
        self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']

        self.embed_dim = args.FFM_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = args.DEVICE

        self.model = _FieldAwareFactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
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
        log_df.to_csv("FM log.csv", mode="w")
    
    
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
