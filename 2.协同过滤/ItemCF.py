# -*- coding: utf-8 -*-
import os
import sys
import json
print(sys.path)

from surprise import accuracy, Dataset, dump, KNNWithMeans
from surprise.model_selection import train_test_split
from surprise import KNNBaseline

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
# ur以user id为key，以user评论过的所有商品id以及对应评分组成的list为value
# ir以item id为key，以评估过该item的所有用户id以及对应评分组成的list为value
# 代码实现过程中，UserCF中：x表示用户，y表示商品；yr就是dataset中的ir, xr就是dataset中的ur
# 代码实现过程中，ItemCF中：x表示物品，y表示用户；yr就是dataset中的ur, xr就是dataset中的ir
# https://surprise.readthedocs.io/en/stable/prediction_algorithms.html#similarity-measures-configuration
sim_options = {
    # "name": "msd_python",
    "name": "msd",
    "user_based": False,  # True表示UserCF、False表示ItemCF
}
bsl_options = {
    'method': 'sgd',
    'n_epochs': 20,
    'learning_rate': 0.005,
    'reg': 0.02
}
# algo = KNNBasic(k=10, min_k=1, sim_options=sim_options)
# algo = KNNWithMeans(k=10, min_k=1, sim_options=sim_options)
algo = KNNBaseline(k=10, min_k=1, sim_options=sim_options, bsl_options=bsl_options)

# 4. 模型训练
# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)  # 训练

# 5. 模型评估
# Then compute RMSE
predictions = algo.test(testset)  # 预测
accuracy.rmse(predictions)  # 评估
accuracy.mse(predictions)  # 评估
accuracy.fcp(predictions)  # 评估

# 6. 模型持久化
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

# 8. 业务逻辑代码
# a. 获取的是每个用户对应的K个推荐商品列表
K=10
rec_path1 = 'output/ItemCF_REC.csv'
all_users_iuid =trainset.ur.keys()
all_items_iiid = trainset.ir.keys()
ur=trainset.ur.copy()

with open(rec_path1,"w") as f:
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
print()
# b. 针对每个商品获取K个相似商品
sim_item = algo.sim
rec_path2 = 'output/ItemCF_REC_sim.csv'

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
print("ItemCF运行结束。。。。。。")