import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
from ._models import RMSELoss, FeaturesEmbedding, FactorizationMachine_v


class CNN_Base(nn.Module):
    def __init__(self, ):
        super(CNN_Base, self).__init__()
        # layer가 되게 작네 그냥 여기는 pre-trained 넣는게 개이득일듯
        self.cnn_layer = nn.Sequential(
                                        nn.Conv2d(3, 6, kernel_size=3, stride=2, padding=1),
                                        nn.BatchNorm2d(6),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=3, stride=2),
                                        nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=1),
                                        nn.BatchNorm2d(12),
                                        nn.ReLU(),
                                        )
    def forward(self, x):
        x = self.cnn_layer(x)
        x = x.view(-1, 12 * 4 * 4)
        return x


class _CNN_FM(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, latent_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.cnn = CNN_Base()
        # 오호 FM을 써버리는군 여기서 ㅇㅇ
        # 그 FM_v는 여기서 쓰는거였구나 ㅋㅋㅋㅋㅋㅋㅋ
        self.fm = FactorizationMachine_v(
                                         input_dim=(embed_dim * 2) + (12 * 4 * 4),
                                         latent_dim=latent_dim,
                                         )

    def forward(self, x):
        # 오호
        # x == fields  == ([data['user_isbn_vector'].to(self.device)] OR [data['user_isbn_vector'].to(self.device),
        #                   data['img_vector'].to(self.device)])
        user_isbn_vector, img_vector = x[0], x[1]
        # [data['user_isbn_vector'].to(self.device) 임 -> 뭔지는 아직 모름
        user_isbn_feature = self.embedding(user_isbn_vector)
        # data['img_vector'].to(self.device) 임 -> 아마도 이미지? cnn에 넣으니까?
        img_feature = self.cnn(img_vector)
        # 대충 모델 좍펴서 그냥 바로 concatenate해버리나봄
        feature_vector = torch.cat([
                                    user_isbn_feature.view(-1, user_isbn_feature.size(1) * user_isbn_feature.size(2)),
                                    img_feature
                                    ], dim=1)
        # 결국, FM에서 context 부분을 CNN feature로 좀 추가했나봄
        output = self.fm(feature_vector)
        return output.squeeze(1)


class CNN_FM:
    def __init__(self, args, data):
        super().__init__()
        self.device = args.DEVICE
        self.model = _CNN_FM(
                            np.array([len(data['user2idx']), len(data['isbn2idx'])], dtype=np.uint32),
                            args.CNN_FM_EMBED_DIM,
                            args.CNN_FM_LATENT_DIM
                            ).to(self.device)
        self.optimizer =  torch.optim.Adam(self.model.parameters(), lr=args.LR)
        self.train_data_loader = data['train_dataloader']
        self.valid_data_loader = data['valid_dataloader']
        self.criterion = RMSELoss()
        self.epochs = args.EPOCHS
        self.model_name = 'image_model'


    def train(self):
        minimum_loss = 999999999
        loss_list = []
        tk0 = tqdm.tqdm(range(self.epochs), smoothing=0, mininterval=1.0)
        for epoch in tk0:
            self.model.train()
            total_loss = 0
            n = 0
            for i, data in enumerate(self.train_data_loader):
                # 이거 뭔데? len(data) == 2 or 3?? 대체 data는 어떤 놈일까 -> 추정: 이거 그러면 img가 없는 책이 있는 경우가 존재하는 경우를 고려했나봄
                # fields: 위의 _CNN_FM의 foward에 들어가는 data임. 즉,
                # fields == x <==>  user_isbn_vector, img_vector = x[0], x[1] 이거 임.
                
                # 아니 근데 이거 이러면 fields 형식이 다른데 self.model(fields)에 바로 넣어도 에러가 안 나오나보네? 뭐지
                if len(data) == 2:
                    fields, target = [data['user_isbn_vector'].to(self.device)], \
                                    data['label'].to(self.device)
                else:
                    fields, target = [data['user_isbn_vector'].to(self.device), data['img_vector'].to(self.device)], \
                                    data['label'].to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n += 1
            
            # validation 시작
            self.model.eval()
            val_total_loss = 0
            val_n = 0
            for i, data in enumerate(self.valid_data_loader):
                if len(data) == 2:
                    fields, target = [data['user_isbn_vector'].to(self.device)], data['label'].to(self.device)
                else:
                    fields, target = [data['user_isbn_vector'].to(self.device), data['img_vector'].to(self.device)], data['label'].to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                # self.model.zero_grad()
                # loss.backward()
                # self.optimizer.step()
                val_total_loss += loss.item()
                val_n += 1
            if minimum_loss > (val_total_loss/val_n):
                minimum_loss = (val_total_loss/val_n)
                if not os.path.exists('./models'):
                    os.makedirs('./models')
                torch.save(self.model.state_dict(), './models/{}.pt'.format(self.model_name))
                loss_list.append([epoch, total_loss/n, val_total_loss/val_n, 'Model saved'])
            else:
                loss_list.append([epoch, total_loss/n, val_total_loss/val_n, 'None'])
            tk0.set_postfix(train_loss=total_loss/n, valid_loss=val_total_loss/val_n)


    def predict(self, test_data_loader):
        self.model.eval()
        self.model.load_state_dict(torch.load('./models/{}.pt'.format(self.model_name)))
        targets, predicts = list(), list()
        with torch.no_grad():
            for data in test_data_loader:
                if len(data) == 2:
                    fields, target = [data['user_isbn_vector'].to(self.device)], data['label'].to(self.device)
                else:
                    fields, target = [data['user_isbn_vector'].to(self.device), data['img_vector'].to(self.device)], data['label'].to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        return predicts
