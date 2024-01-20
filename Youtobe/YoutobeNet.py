import torch
import torch.nn as nn
import numpy as np
class Deep_NetWork(nn.Module):
    '''Deep_NetWork'''
    def __init__(self, input_dim, hidden_dims, dropout, output_layer=False):
        super().__init__()
        self.layers = list()
        for embed_dim in hidden_dims:
            self.layers.append(nn.Linear(input_dim, embed_dim))
            self.layers.append(nn.BatchNorm1d(embed_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            self.layers.append(nn.Linear(input_dim, output_layer,bias=False))
        self.mlp = nn.Sequential(*self.layers)
    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim*field_dim)``
        :return self.mlp(x):float torchsize ``(batch_size, 1)``
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

class YoutobeNet(nn.Module):
    def __init__(self,field_dims,embed_dim=8,dropout=0,hidden_size=[1024,512,256]):
        super(YoutobeNet,self).__init__()
        input_dim = len(field_dims)*embed_dim
        self.embed = FeaturesEmbedding(field_dims, embed_dim)
        self.mlp = Deep_NetWork(input_dim, hidden_size, dropout, output_layer=6)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self,x):
        x = self.embed(x)
        x = torch.reshape(x,[x.shape[0],-1])
        x = self.mlp(x)
        return self.softmax(x)
    def get_item_vector(self):
        return self.mlp.layers[-1].weight


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
            #构建样本
            if i ==j:
                data_pos = X_user[i]+X_items[j]+[j,]
                X.append(data_pos)
    X_y=torch.tensor(X)
    X=X_y[:,:-1]
    y=torch.zeros((X.shape[0],len(X_items)))
    y_ =  X_y[:,-1]
    for i in range(len(y_)):
        y[i,y_[i]]=1

    #用户和商品特征的类别数：id:6,性别:2,职业：20,家庭地址:10,商品id:6、商品所属店铺id:20、商品类别id:10
    field_featrues = torch.tensor([6,2,20,10,6,20,10])
    model=YoutobeNet(field_featrues)
    #交叉熵损失
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(),lr=0.01)
    for i in range(1000):
        y_pred = model(X)
        loss = loss_fn(y_pred,y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss)
    item_vector = model.get_item_vector()
    with open('item_vector.txt','w') as f:
        f.write("{\n")
        for i in range(len(X_items)):
            f.writelines(f"'{X_items[i][0]}':{item_vector[i]}")
            f.write("\n")
        f.write("}")
if __name__ == '__main__':
    t0()