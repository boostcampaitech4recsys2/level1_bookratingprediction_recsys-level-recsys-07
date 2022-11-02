import numpy as np

import torch
import torch.nn as nn

def rmse(real: list, predict: list) -> float:
    pred = np.array(predict)
    return np.sqrt(np.mean((real-pred) ** 2))


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y)+self.eps)
        return loss


# self.model = \
# _FactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
"""
FM 논문 공식을 단순하게 구현함 -> latent_dim 활용 안함 and nn.Linear 사용 안함
return: 상수값
"""
class FactorizationMachine(nn.Module):

    def __init__(self, reduce_sum:bool=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x: torch.Tensor):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


"""
FM 논문 공식을 그대로 구현함 -> latent_dim 활용 함
return: 벡터
"""
class FactorizationMachine_v(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.v = nn.Parameter(torch.rand(input_dim, latent_dim), requires_grad = True)
        self.linear = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        linear = self.linear(x)
        square_of_sum = torch.mm(x, self.v) ** 2
        sum_of_square = torch.mm(x ** 2, self.v ** 2)
        pair_interactions = torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        output = linear + (0.5 * pair_interactions)
        return output


# 다른 모델들도 동일하게 쓴다. (FFM만 이거를 그대로 안 씀)
# 첫번째 feature의 라벨만큼 더해줘서 x가 multi-hot이 된걸로 이해했음
class FeaturesEmbedding(nn.Module):

    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

# FM, FFM만 사용
# nn.linear효과와 같음
class FeaturesLinear(nn.Module):

    def __init__(self, field_dims: np.ndarray, output_dim: int=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


# (self.linear(x) + self.fm(self.embedding(x))).squeeze(1)
# linear를 더해줘야하나보네?
class _FactorizationMachineModel(nn.Module):

    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = self.linear(x) + self.fm(self.embedding(x))
        # return torch.sigmoid(x.squeeze(1))
        return x.squeeze(1)


class FieldAwareFactorizationMachine(nn.Module):

    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.num_fields = len(field_dims)
        
        # 여러 nn.Embedding 층을 쌓음
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)
        ])
        # 이거는 다른 모델들이랑 동일함
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets, dtype= np.long).unsqueeze(0)
        
        # 여러 nn.Embedding 층에 통과시킴
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]
        # field간 곱연산의 결과들을 ix에 저장함
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(xs[j][:, i] * xs[i][:, j])
        # stack으로 합치기
        ix = torch.stack(ix, dim=1)
        return ix

# 논문 구현 부분인가?
class _FieldAwareFactorizationMachineModel(nn.Module):

    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.ffm = FieldAwareFactorizationMachine(field_dims, embed_dim)

    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        ffm_term = torch.sum(torch.sum(self.ffm(x), dim=1), dim=1, keepdim=True)
        x = self.linear(x) + ffm_term
        # return torch.sigmoid(x.squeeze(1))
        return x.squeeze(1)

# input으로 받은 embed_dims에 맞추서 MLP layer 쌓으
class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)
        self.last_mlp_layer_size = embed_dims[-1]
    
    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        identity = x
        breakpoint()
        return self.relu(self.mlp(x) + identity.view(-1, self.last_mlp_layer_size))
        # return self.mlp(x)

class _NeuralCollaborativeFiltering(nn.Module):

    def __init__(self, field_dims, field_idx_dict, embed_dim, mlp_dims, dropout, batch_size):
        super().__init__()
        self.field_idx_dict = field_idx_dict
        """
        FeaturesEmbedding(field_dims, embed_dim)
            field_dims: [유저 전체 수, 아이템 전체 수] == np.array([len(user2idx), len(isbn2idx)], dtype=np.uint32)
            embed_dim: args.NCF_EMBED_DIM -> user_vector의 가로 길이
                몇 k차원으로 임베딩할 것인가
            self.embedding 통과 후: 2 * embed_dim으로 변환
            [ 68069 149570], 16
            68069 149570 -> 20만
        """
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.fc = torch.nn.Linear(embed_dim, 1)
        self.last_mlp_layer = mlp_dims[-1]
        self.batch_size = batch_size

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_user_fields)``
        """
        
        # print(x.shape) torch.Size([1024, (2 + context_feature 수)])
        # user & item vector -> 임베딩
        x = self.embedding(x)
        # print(x.shape) torch.Size([1024, (2 + context_feature 수), 16])
        
        # user_x = x[:, self.user_field_idx].squeeze(1)
        # item_x = x[:, self.item_field_idx].squeeze(1)
        # gmf = user_x * item_x
        
        gmf = x[:, self.field_idx_dict['user_id']].squeeze(1)
        for field_name, field_idx in self.field_idx_dict.items():
            if field_name == 'user_id':
                continue
            tmp = x[:, self.field_idx_dict[field_name]].squeeze(1)
            gmf = (gmf * tmp)
        
        # MLP 통과
        # x.shape: torch.Size([1024, (2 + context_feature 수), 16])
        # x.view(-1, self.embed_output_dim).shape: torch.Size([1024, 32])
        # self.mlp(x.view(-1, self.embed_output_dim)).shape: ([1024, (2 + context_feature 수)56])
        # breakpoint()
        # # shape '[-1, 176]' is invalid for input of size 163840
        x = x.view(-1, self.embed_output_dim)
        # breakpoint()
        # x = x.view(self.batch_size ,-1, self.last_mlp_layer)
        # breakpoint()
        x = self.mlp(x)
        # breakpoint()
        x = torch.cat([gmf, x], dim=1)
        # breakpoint()
        # x = gmf
        # breakpoint()
        x = self.fc(x).squeeze(1)
        return x

# 더 단순하네?
class _WideAndDeepModel(nn.Module):

    def __init__(self, field_dims: np.ndarray, embed_dim: int, mlp_dims: tuple, dropout: float):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return x.squeeze(1)

class CrossNetwork(nn.Module):

    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        # x0는 유지함
        x0 = x
        # nn.linear
        # b가 좀 의문이네, 그냥 nn.linear하면 bias항 안에 포함된거 아님?
        for i in range(self.num_layers):
            # xw = x_l^T * w_l 부분
            xw = self.w[i](x)
            # x_(l+1) = x_0 * x_l^T * w_l + b_l + x_l
            x = x0 * xw + self.b[i] + x
        return x

class _DeepCrossNetworkModel(nn.Module):
    """
    A pytorch implementation of Deep & Cross Network.
    Reference:
        R Wang, et al. Deep & Cross Network for Ad Click Predictions, 2017.
    """

    def __init__(self, field_dims: np.ndarray, embed_dim: int, num_layers: int, mlp_dims: tuple, dropout: float):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.cd_linear = nn.Linear(mlp_dims[0], 1, bias=False)

    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        x_l1 = self.cn(embed_x)
        x_out = self.mlp(x_l1)
        p = self.cd_linear(x_out)
        return p.squeeze(1)
