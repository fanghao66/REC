import torch.nn as nn
import numpy as np
import torch
class Deep_NetWork(nn.Module):
    '''Deep_NetWork'''
    def __init__(self, input_dim, hidden_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in hidden_dims:
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
        :return self.mlp(x):float torchsize ``(batch_size, 1)``
        """
        return self.mlp(x)
class Linear_Work(nn.Module):
    def __init__(self,field_dims):
        super(Linear_Work, self).__init__()
        self.embed = nn.Embedding(sum(field_dims),1)
        self.bias = nn.Parameter(torch.zeros(size=(1,)))
    def forward(self,x):
        '''
        :param x: int torchsize ''[N,field_dims]''
        :return self.linear(x) :float torchsize:[N,1]
        '''
        x = self.embed(x)#[N,field_dims,1]
        x = torch.sum(x,dim=(1,2))+self.bias
        return torch.reshape(x,(-1,1))
class CIN_NetWork(nn.Module):
    def __init__(self,field_dims,embed_dims,CIN_layer_num,Hk=10):
        super(CIN_NetWork, self).__init__()
        self.embed = nn.Embedding(sum(field_dims),embed_dims)
        self.linear = nn.Linear(CIN_layer_num*Hk,1)

        self.cin_net = []
        M = len(field_dims)
        H = M
        for i in range(CIN_layer_num):
            self.cin_net.append(CIN_layer(H,M,Hk))
            H = Hk
    def forward(self,x):
        '''
        :param x: int torchsize``(bathch_size,field_dims)
        :return:
        '''
        x = self.embed(x)#(bathch_size,field_dims,embed_dims)
        x_ = x
        x_list = []
        for net in self.cin_net:
            x_ = net(x_,x)#(bathch_size,Hk,embed_dims)
            x_list.append(x_)
        x = torch.cat([torch.sum(i,dim=-1) for i in x_list],dim=-1)#(bathch_size,Hk*cin_layer_num)
        return self.linear(x)



class CIN_layer(nn.Module):
    def __init__(self,H,M,H_next=3):
        super(CIN_layer, self).__init__()
        self.weight = []
        for i in range(H_next):
            self.weight.append(nn.Linear(H*M,1,bias=False))
    def forward(self,X_l,X_0):
        '''
        :param X_l: float torchsize ``(bathch_size,Hk,embed_dims)``
        :param X_0: float torchsize ``(bathch_size,M,embed_dims)``
        :return:
        '''
        X_l = torch.permute(X_l,[0,2,1])
        X_l = torch.unsqueeze(X_l,dim=-1)#(bathch_size,embed_dims,Hk,1)
        X_0 = torch.permute(X_0, [0, 2, 1])
        X_0 = torch.unsqueeze(X_0, dim=-2)#(bathch_size,embed_dims,1,M)
        X_ll = torch.matmul(X_l, X_0)#(bathch_size,embed_dims,Hk,M)
        X_ll = torch.reshape(X_ll,(X_ll.shape[0],X_ll.shape[1],-1))#(bathch_size,embed_dims,Hk*M)
        X_ll_list = []
        for linear in self.weight:
            X_ll_ = linear(X_ll)#(bathch_size,embed_dims,1)
            X_ll_list.append(X_ll_)
        X_ll = torch.cat(X_ll_list,dim=-1)#(bathch_size,embed_dims,H_next)
        return torch.permute(X_ll,(0,2,1))#(bathch_size,H_next,embed_dims)
class xDeepFM(nn.Module):
    def __init__(self,field_dims,embed_dims,deepnet_hiddendims,CIN_layer_num=3):
        super(xDeepFM, self).__init__()
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        self.embed_inputdims = len(field_dims)*embed_dims

        self.linear = Linear_Work(field_dims=field_dims)
        self.CIN = CIN_NetWork(field_dims,embed_dims,CIN_layer_num)
        self.mlp = Deep_NetWork(self.embed_inputdims,deepnet_hiddendims,dropout=0.2)

        self.sigmoid = nn.Sigmoid()
        pass
    def forward(self,x):
        '''
        :param x: int torchsize ``(batch_size,field_dims)``
        :return:
        '''
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        #1.linear
        x_1 = self.linear(x)#[batch_size,1]
        #2.CIN
        x_2 = self.CIN(x)#[batch_size,1]
        #3.mlp
        x = self.CIN.embed(x)#[batch_size,field_dims,embed_dims]
        x = torch.reshape(x,(x.shape[0],-1))#[batch_size,field_dims*embed_dims]
        x_3 = self.mlp(x)#[batch_size,1]
        return self.sigmoid(1/3*(x_1+x_2+x_3))

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

    model=xDeepFM(field_dims=field_featrues,embed_dims=8,deepnet_hiddendims=[1024,512,256,1],CIN_layer_num=3)
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
if __name__ == '__main__':
    t0()
