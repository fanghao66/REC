import torch
import torch.nn as nn
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
    def __init__(self,field_featrues,vector_dim):
        super(FM_Model, self).__init__()
        self.field_features=field_featrues

        self.linear_layer = Linear_Layer(field_featrues)
        self.fm = FactorizationMachine(field_featrues,vector_dim)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        '''
        x:[N,feat_num]
        '''
        #1.对原始数据做编码，一个特征对应一个序号
        N, _ = x.shape
        x_ = []
        for i in range(N):
            _ = x[i]
            _ = [(_[i] + sum(self.field_features[:i])).item() for i in range(len(_))]
            x_.append(_)
        x = torch.tensor(x_)
        self.item = x[:,-3:]
        #2.一阶特征交叉
        y_linear = self.linear_layer(x)
        #3.二阶特征交叉
        y_fm = torch.reshape(self.fm(x),(-1,1))
        return self.sigmoid(y_linear+y_fm)
    def get_Item_vector(self):
        '''
        Items:[N,field_num_Item]
        '''
        item=torch.unique(self.item,dim=0)
        item_id =item[:,0]-torch.min(item[:,0])
        item_vector = self.fm.embed(item)#[N,3,vector]
        item_vector = torch.sum(item_vector,dim=1)
        return item_id,item_vector
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
    #用户和商品特征的类别数：id:6,性别:2,职业：20,家庭地址:10,商品id:6、商品所属店铺id:20、商品类别id:10
    field_featrues = torch.tensor([6,2,20,10,6,20,10])
    fm=FM_Model(field_featrues,8)
    loss_fn = nn.BCELoss()
    opt = torch.optim.AdamW(fm.parameters(),lr=0.01)
    for i in range(1000):
        y_pred = fm(X)
        loss = loss_fn(y_pred,y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss)
    item_id,item_vector = fm.get_Item_vector()
    with open('item_vector.txt','w') as f:
        f.write("{\n")
        for i in range(len(item_id)):
            f.writelines(f"'{item_id[i]}':{item_vector[i]}")
            f.write("\n")
        f.write("}")
if __name__=="__main__":
    t0()