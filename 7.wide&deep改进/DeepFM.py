import torch.nn as nn
import numpy as np
import torch
class FeaturesEmbedding(nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        #将用单个类别序号表示的x，用整个数据特征序号表示，方便embedding
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

class Deep_NetWork(nn.Module):
    '''Deep_NetWork'''
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim*field_dim)``
        """
        return self.mlp(x)

class Linear_Layer(nn.Module):
    '''做一阶特征交叉'''
    def __init__(self,field_features,output_dim=1):
        super(Linear_Layer, self).__init__()
        self.field_features = field_features
        self.embed = nn.Embedding(sum(field_features),1)
    def forward(self,x):
        '''
        x:[N,feat_num]
        '''
        x = self.embed(x)
        x = torch.sum(x,dim=1)
        return x

class FactorizationMachine(nn.Module):
    '''做二阶特征交叉'''
    def __init__(self,field_featrues, vector_dim):
        super(FactorizationMachine, self).__init__()
        self.embed = nn.Embedding(sum(field_featrues), vector_dim)
    def forward(self,x):
        '''
        x:torchsize:[N,feat_num]
        '''
        x = self.embed(x)#[N,feat_num,vector_dim]
        square_of_sum = torch.sum(x,dim=1)**2#[N,vector_dim]
        sum_of_square = torch.sum(x**2,dim=1)#[N,vector_dim]
        x=torch.sum((square_of_sum-sum_of_square),dim=1)#[N,]
        return x
class FM_Model(nn.Module):
    def __init__(self,field_featrues,embed_dim):
        super(FM_Model, self).__init__()
        self.field_features=field_featrues
        self.linear_layer = Linear_Layer(field_featrues)
        self.fm = FactorizationMachine(field_featrues,embed_dim)
    def forward(self,x):
        '''
        x:[N,field_dim]
        '''
        #1.一阶特征交叉
        y_linear = self.linear_layer(x)
        #2.二阶特征交叉
        y_fm = torch.reshape(self.fm(x),(-1,1))
        return y_linear+y_fm
class DeepFM(nn.Module):
    def __init__(self,field_dims,embed_dim, dropout, mlp_dims):
        super(DeepFM,self).__init__()
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = Deep_NetWork(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.fm = FM_Model(field_dims, embed_dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        # 1.对原始数据做编码，一个特征对应一个序号
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        # 2.FM层
        x_1 = self.fm(x)#[N,1]
        # 3.Deep Network
        x = self.fm.fm.embed(x)
        x = torch.reshape(x,[x.shape[0], -1])
        x_2 = self.mlp(x)#[N,1]
        return self.sigmoid(0.5*(x_1+x_2))


def t0():
    X_user = [
        [0, 1, 14, 8],
        [1, 0, 10, 7],
        [2, 0, 12, 6],
        [3, 1, 1, 5],
        [4, 0, 4, 4],
        [5, 1, 10, 6],
    ]
    X_items = [
        [0, 1, 4],
        [1, 10, 1],
        [2, 11, 2],
        [3, 3, 1],
        [4, 9, 4],
        [5, 3, 1]
    ]
    X = []
    # 构建数据集
    for i in range(len(X_user)):
        for j in range(len(X_items)):
            # g构建正样本
            if i == j:
                data_pos = X_user[i] + X_items[j] + [1, ]
                X.append(data_pos)
            # 构建负样本
            else:
                data_neg = X_user[i] + X_items[j] + [0, ]
                X.append(data_neg)
    X_y = torch.tensor(X)
    X = X_y[:, :-1]
    y = torch.reshape(X_y[:, -1], (-1, 1)).to(torch.float32)
    field_featrues = torch.tensor([6, 2, 20, 10, 6, 20, 10])
    model = DeepFM(field_dims=field_featrues ,embed_dim=8, dropout=0.8, mlp_dims=[1024,512,256,1])
    loss_fn = nn.BCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    for i in range(1000):
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss)
    print()
t0()