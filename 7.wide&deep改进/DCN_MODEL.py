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
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

class CrossNetwork(nn.Module):
    """Cross Network    """
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = [
            nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ]
        self.b = [
            nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ]

    def forward(self, x):
        """
        :param x: torchsize '[batch_size,embed_dim*field_dim]'
        """
        x0 = x
        for i in range(self.num_layers):
            #x_{l+1} = x_0*(x_l*w_l) + b_l + x_l
            xl_wl = self.w[i](x)
            x = x0 * xl_wl + self.b[i] + x
        return x
class DeepCrossNetworkModel(nn.Module):
    """ Deep & Cross Network """

    def __init__(self, field_dims, embed_dim, num_layers, mlp_dims, dropout):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        #全连接的时候每个属性对应一个参数
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        self.mlp = Deep_NetWork(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.linear = torch.nn.Linear(mlp_dims[-1] + self.embed_output_dim, 1)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        x_l1 = self.cn(embed_x)
        h_l2 = self.mlp(embed_x)
        x_stack = torch.cat([x_l1, h_l2], dim=1)
        p = self.linear(x_stack)
        return torch.sigmoid(p.squeeze(1))

def t0():
    X_user = [
        [0,1,14,8],
        [1,0,10,7],
        [2,0,12,6],
        [3,1,1,5],
        [4,0,4,4],
        [5,1,10,6],
    ]
    X_items= [
        [0,1,4],
        [1,10,1],
        [2,11,2],
        [3,3,1],
        [4,9,4],
        [5,3,1]
    ]
    X=[]
    #构建数据集
    for i in range(len(X_user)):
        for j in range(len(X_items)) :
            #g构建正样本
            if i ==j:
                data_pos = X_user[i]+X_items[j]+[1,]
                X.append(data_pos)
            #构建负样本
            else:
                data_neg = X_user[i]+X_items[j]+[0,]
                X.append(data_neg)
    X_y=torch.tensor(X)
    X=X_y[:,:-1]
    y=torch.reshape(X_y[:,-1],(-1,1)).to(torch.float32)
    field_featrues = torch.tensor([6, 2, 20, 10, 6, 20, 10])
    model=DeepCrossNetworkModel(field_featrues,8,3,[1024,512,256],0.8)
    y_pred=model(X)
    print(y_pred)
if __name__ == '__main__':
    t0()

