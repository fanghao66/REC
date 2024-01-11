# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
print(sys.path)

from surprise import accuracy, Dataset, dump, SVD
from surprise.model_selection import train_test_split

# 1. 加载数据
# Load the movielens-100k dataset (download it if needed)
# 加载ml-100k的数据集，如果不存在，会从网络上下载: C:\Users\19410\.surprise_data
data = Dataset.load_builtin("ml-100k")

# 2. 数据转换
# sample random trainset and testset 数据分割成75%的训练数据集 + 25%的测试数据集
# 训练数据的转换：外部id和内部id互相转换的元数据、每个用户的评论商品列表、每个商品评论的用户列表....
# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=0.25)

# 3. 构建模型对象
# https://surprise.readthedocs.io/en/stable/matrix_factorization.html
algo = SVD(
    n_factors=10,
    n_epochs=100,
    biased=True
)

# 4. 模型训练
# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)  # 训练

# 5. 模型评估
# Then compute RMSE
predictions = algo.test(testset)  # 预测
accuracy.rmse(predictions)  # 评估
accuracy.mse(predictions)  # 评估
accuracy.fcp(predictions)  # 评估

# # 6. 模型持久化
# output_dir = f"output/{algo.__class__.__name__}_{'user_cf' if algo.sim_options.get('user_based', True) else 'item_cf'}"
# os.makedirs(output_dir, exist_ok=True)
# dump.dump(f"{output_dir}/model.pkl", predictions=None, algo=algo)
# _, algo = dump.load(f"{output_dir}/model.pkl")

# 7.wide&deep改进. 模型的预测
print("=" * 100)
y_ = algo.predict(uid="196", iid="224", r_ui=None, clip=True)
print(y_)
print(f"预测评分:{y_.est:.2f}")
y_ = algo.predict(uid="196", iid="224", r_ui=3.5, clip=True)
print(y_)
print(f"预测评分:{y_.est:.2f}")
y_ = algo.predict(uid="196", iid="224", r_ui=3.5, clip=True)
print(y_)
print(f"预测评分:{y_.est:.2f}")

print("=" * 100)
print(f"用户评分偏移向量大小:{algo.bu.shape}")
print(f"物品评分偏移向量大小:{algo.bi.shape}")
print(f"用户-潜在因子矩阵大小:{algo.pu.shape}")
print(f"物品-潜在因子矩阵大小:{algo.qi.shape}")

print("=" * 100)
# 8. 业务逻辑代码
# a. 获取的是每个用户对应的K个推荐商品列表
K=10
rec_path = 'output/SVD_REC.csv'
all_users_iuid =trainset.ur.keys()
all_items_iiid = trainset.ir.keys()
ur=trainset.ur.copy()

with open(rec_path,"w") as f:
    f.write("{\n")
    for iuid in all_users_iuid:
        uid = trainset.to_raw_uid(iuid)
        REC=[]
        uid_iiid_do= [i[0] for i in trainset.ur[iuid]]
        for iiid in all_items_iiid:
            iid = trainset.to_raw_iid(iiid)
            r = algo.predict(uid, iid).est
            if iiid not in uid_iiid_do:
                REC.append((iid,r))
        REC=sorted(REC,key=lambda x:x[1],reverse=True)[:K]
        f.writelines(f"'{uid}':{REC}")
        f.write("\n")
    f.write("}")
# b. 针对每个商品获取K个相似商品
def cal_Item_sim(x):
    x = x / np.linalg.norm(x, axis=1).reshape(-1, 1)
    return np.dot(x,x.T)
Item_factor = algo.qi
rec_path2 = 'output/SVD_REC_sim.csv'
sim_item = cal_Item_sim(Item_factor)
with open(rec_path2,"w") as f:
    f.write("{\n")
    for iiid in range(sim_item.shape[0]):
        iid = trainset.to_raw_iid(iiid)
        sim_1=sim_item[iiid]
        sim_2=sim_1>0.4
        sim_2[iiid]=False
        rec_list = [(trainset.to_raw_iid(i),sim_1[i]) for i,j in enumerate(sim_2) if j]
        f.writelines(f"'{iid}':{rec_list}")
        f.write("\n")
    f.write("}")
print("SVD运行结束。。。。。。")