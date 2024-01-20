"""
DSSM 双塔模型
"""

import sys
import torch
import torch.nn as nn
import numpy as np
class DSSM(nn.Module):
    def __init__(self, field_dim_user, field_dim_item, user_dnn_size=(300,300, 128),
                 item_dnn_size=(300,300, 128), dropout=0.0, embed_dim=8):
        super().__init__()

        self.user_dnn_size = user_dnn_size
        self.item_dnn_size = item_dnn_size
        self.dropout = dropout
        self.input_item = len(field_dim_item)*embed_dim
        self.input_user = len(field_dim_user)*embed_dim

        #用户embedding
        self.embed_user = FeaturesEmbedding(field_dim_user, embed_dim)
        #商品embedding
        self.embed_item = FeaturesEmbedding(field_dim_item,embed_dim)

        # 用户侧
        self.user_tower = Tower(self.input_user, user_dnn_size, dropout, output_layer=False)
        #商品侧
        self.item_tower = Tower(self.input_item, item_dnn_size, dropout, output_layer=False)

    def forward(self, user_feat, item_feat):
        user_feat = torch.reshape(self.embed_user(user_feat),(user_feat.shape[0],-1))
        item_feat = torch.reshape(self.embed_item(item_feat),(item_feat.shape[0],-1))

        user_feat = self.user_tower(user_feat)#[N,hidden_dim]
        item_feat = self.item_tower(item_feat)

        user_feat = user_feat/torch.sqrt(torch.sum(user_feat**2,dim=-1,keepdim=True))
        item_feat = item_feat / torch.sqrt(torch.sum(item_feat ** 2, dim=-1, keepdim=True))
        return user_feat,item_feat
    def get_item_vector(self,item_feat):
        item_feat = torch.reshape(self.embed_item(item_feat), (item_feat.shape[0], -1))
        item_feat = self.item_tower(item_feat)
        item_feat = item_feat / torch.sqrt(torch.sum(item_feat ** 2, dim=-1, keepdim=True))
        return item_feat
class Tower(nn.Module):
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
        """
        return self.mlp(x)
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
                data_neg = X_user[i] + X_items[j] + [-1, ]
                X.append(data_neg)
    X_y = torch.tensor(X)
    X = X_y[:, :-1]
    y = X_y[:, -1]
    field_dim_user = torch.tensor([6, 2, 20, 10])
    field_dim_item = torch.tensor([6, 20, 10])
    model = DSSM(field_dim_user, field_dim_item, user_dnn_size=(300,300, 128),
                 item_dnn_size=(300,300, 128), dropout=0.0, embed_dim=8)
    #利用余弦损失函数
    loss_fn = nn.CosineEmbeddingLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    for i in range(1000):
        user_vector,item_vector_ = model(X[:,:4],X[:,4:])
        loss = loss_fn(user_vector,item_vector_, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss)
    item_vector = model.get_item_vector(torch.tensor(X_items))
    with open('item_vector.txt','w') as f:
        f.write("{\n")
        for i in range(len(item_vector)):
            f.writelines(f"'{X_items[i][0]}':{item_vector[i]}")
            f.write("\n")
        f.write("}")
    print()
if __name__ == '__main__':
    t0()