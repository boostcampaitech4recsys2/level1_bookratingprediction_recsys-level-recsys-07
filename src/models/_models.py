import numpy as np

import torch
import torch.nn as nn

def rmse(real: list, predict: list) -> float:
    pred = np.array(predict)
    return np.sqrt(np.mean((real-pred) ** 2))

def activation_layer(act_name):
    '''Select activation layer by its name
    Parameter
        act_name: String value or nn.Module, name of activation function
    Return
        act_layer: Activation layer
    '''
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'linear':
            act_layer = nn.Identity()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError

    return act_layer


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y)+self.eps)
        return loss


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
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)
        ])
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets, dtype= np.long).unsqueeze(0)
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(xs[j][:, i] * xs[i][:, j])
        ix = torch.stack(ix, dim=1)
        return ix

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

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

class _NeuralCollaborativeFiltering(nn.Module):

    def __init__(self, field_dims, field_idx_dict, embed_dim, mlp_dims, dropout):
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
        self.fc = torch.nn.Linear(mlp_dims[-1] + embed_dim, 1)

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
        # shape '[-1, 176]' is invalid for input of size 163840
        x = x.view(-1, self.embed_output_dim)
        # breakpoint()
        x = self.mlp(x)
        # breakpoint()
        x = torch.cat([gmf, x], dim=1)
        x = self.fc(x).squeeze(1)
        return x

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
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x

class _DeepCrossNetworkModel(nn.Module):
    """
    A pytorch implementation of Deep & Cross Network.
    Reference:
        R Wang, et al. Deep & Cross Network for Ad Click Predictions, 2017.
    """

    def __init__(self, n_features: int, field_idx: np.ndarray, field_dims: np.ndarray, embed_dim: int, num_layers: int, mlp_dims: tuple, dropout: float):
        super().__init__()
        self.field_idx = field_idx
        self.dense_idx = np.delete(np.arange(n_features), self.field_idx)
        # self.embeddings = torch.nn.ModuleList([
        #     torch.nn.Embedding(field_dim, embed_dim) for field_dim, embed_dim in zip(field_dims, embed_dims)
        # ])
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = embed_dim*len(field_dims)
        self.dimreduction = torch.nn.Linear(512, 32)
        self.nn_input_dim = self.embed_output_dim + 32
        self.cn = CrossNetwork(self.nn_input_dim, num_layers)
        self.mlp = MultiLayerPerceptron(self.nn_input_dim, mlp_dims, dropout, output_layer=False)
        self.cd_linear = nn.Linear(self.nn_input_dim+mlp_dims[-1], 1, bias=False)

    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x_continuous = x[:,self.dense_idx]
        x_continuous = self.dimreduction(x_continuous)
        x_sparse = x[:,self.field_idx].long()
        x_embed = self.embedding(x_sparse).view(-1, self.embed_output_dim)
        # embed_x = torch.cat([self.embeddings[i](x[:,idx]) for i, idx in enumerate(self.field_idx)], dim=1)
        x_cat = torch.cat([x_embed, x_continuous], dim=1)
        # embed_x = self.embedding(x_field).view(-1, self.embed_output_dim)
        x_cross = self.cn(x_cat)
        x_dnn = self.mlp(x_cat)
        p = self.cd_linear(torch.cat([x_cross, x_dnn], dim=1))
        
        return p.squeeze(1)

############################# DeepFM
class FMLayer(nn.Module):
    def __init__(self, input_dim):
        '''
        Parameter
            input_dim: Entire dimension of input vector (sparse)
            embed_dim: Factorization dimension
        '''
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def square(self, x):
        return torch.pow(x,2)

    def forward(self, sparse_x, dense_x):
        '''
        Parameter
            sparse_x : Same with `x_multihot` in FieldAwareFM class
                       Float tensor with size "(batch_size, self.input_dim)"
            dense_x  : Similar with `xv` in FFMLayer class. 
                       Float tensors of size "(batch_size, num_fields, embed_dim)"
        
        Return
            y: Float tensor of size "(batch_size)"
        '''
        
        y_linear = self.linear(sparse_x)
        
        square_of_sum = self.square(torch.sum(dense_x, dim=1))
        sum_of_square = torch.sum(self.square(dense_x), dim=1)
        y_pairwise = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1)
        
        y_fm = y_linear.squeeze(1) + y_pairwise

        return y_fm

class DNNLayer(nn.Module):
    '''The Multi Layer Percetron (MLP); Fully-Connected Layer (FC); Deep Neural Network (DNN) with 1-dimensional output
    Parameter
        inputs_dim: Input feature dimension
        hidden_units: List of positive integer, the layer number and units in each layer
        dropout_rate: Float value in [0,1). Fraction of the units to dropout
        activation: Activation function to use
        use_bn: Boolean value. Whether use BatchNormalization before activation
    '''
    def __init__(self, 
                 input_dim, 
                 hidden_units, 
                 dropout_rate=0, 
                 activation='relu', 
                 use_bn=False,
                 **kwargs):
        super().__init__()
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.use_bn = use_bn
        
        layer_size = len(hidden_units)
        hidden_units = [input_dim] + list(hidden_units)

        self.linears = nn.ModuleList([
            nn.Linear(hidden_units[i], hidden_units[i+1]) for i in range(layer_size)
        ])

        if self.use_bn:
            self.bn = nn.ModuleList([
                nn.BatchNorm1d(hidden_units[i+1]) for i in range(layer_size)
            ])

        self.activation_layers = nn.ModuleList([
            activation_layer(activation) for i in range(layer_size)
        ])
        
        self.dnn_linear = nn.Linear(hidden_units[-1], 1, bias=False)
        
        self._initialize_weights()
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        '''
        Parameter
            x: nD tensor of size "(batch_size, ..., input_dim)"
               The most common situation would be a 2D input with shape "(batch_size, input_dim)".
        
        Return
            y: nD tensor of size "(batch_size, ..., 1)"
               For instance, if input x is 2D tensor, the output y would have shape "(batch_size, 1)".
        '''
        deep_input = x
        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)
            if self.use_bn:
                fc = self.bn[i](fc)
            fc = self.activation_layers[i](fc)
            fc = self.dropout(fc)
            deep_input = fc
        
        y_dnn = self.dnn_linear(deep_input)
            
        return y_dnn

class _DeepFM(nn.Module):
    '''The DeepFM architecture
    Parameter
        field_dims: List of field dimensions
        embed_dim: Factorization dimension for dense embedding
        dnn_hidden_units: List of positive integer, the layer number and units in each layer
        dnn_dropout: Float value in [0,1). Fraction of the units to dropout in DNN layer
        dnn_activation: Activation function to use in DNN layer
        dnn_use_bn: Boolean value. Whether use BatchNormalization before activation in DNN layer
    '''
    def __init__(self,
                 field_dims,
                 embed_dim=5,
                 dnn_hidden_units=(64, 32),
                 dnn_dropout=0,
                 dnn_activation='relu', 
                 dnn_use_bn=False,
                 **kwargs):
        super(_DeepFM, self).__init__(**kwargs)
        
        if len(dnn_hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        
        self.input_dim = sum(field_dims)
        self.num_fields = len(field_dims)
        self.encoding_dims = np.concatenate([[0], np.cumsum(field_dims)[:-1]])
        
        self.embedding = nn.ModuleList([
            nn.Embedding(feature_size, embed_dim) for feature_size in field_dims
        ])
        
        
        self.fm = FMLayer(input_dim=self.input_dim)
        self.dnn = DNNLayer(input_dim=(self.num_fields * embed_dim), 
                            hidden_units=dnn_hidden_units, 
                            activation=dnn_activation, 
                            dropout_rate=dnn_dropout, use_bn=dnn_use_bn)
        
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
        
                
    def forward(self, x):
        '''
        Parameter
            x: Long tensor of size "(batch_size, num_fields)"
                sparse_x : Same with `x_multihot` in FieldAwareFM class
                dense_x  : Similar with `xv` in FFMLayer class 
                           List of "num_fields" float tensors of size "(batch_size, embed_dim)"
        Return
            y: Float tensor of size "(batch_size)"
        '''
        sparse_x = x + x.new_tensor(self.encoding_dims).unsqueeze(0)
        sparse_x = torch.zeros(x.size(0), self.input_dim, device=x.device).scatter_(1, x, 1.)
        dense_x = [self.embedding[f](x[:,f]) for f in range(self.num_fields)] 

        y_fm = self.fm(sparse_x, torch.stack(dense_x, dim=1))
        y_dnn = self.dnn(torch.cat(dense_x, dim=1))
        
        
        y = y_fm + y_dnn.squeeze(1)

        return y