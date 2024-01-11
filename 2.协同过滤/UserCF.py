# -*- coding: utf-8 -*-
import os
import sys

print(sys.path)

from surprise import accuracy, Dataset, dump, KNNWithMeans, KNNBaseline
from surprise.model_selection import train_test_split
from surprise import KNNBasic

# 1. 加载数据
data = Dataset.load_builtin("ml-100k")

# 2. 数据转换
trainset, testset = train_test_split(data, test_size=0.25)

# 3. 构建模型对象
sim_options = {
    "name": "msd",
    "user_based": True,  # True表示UserCF、False表示ItemCF
}
bsl_options = {
    'method': 'sgd',
    'n_epochs': 20,
    'learning_rate': 0.005,
    'reg': 0.02
}
algo = KNNBasic(k=10, min_k=1, sim_options=sim_options)
# algo = KNNWithMeans(k=10, min_k=1, sim_options=sim_options)
# algo = KNNBaseline(k=10, min_k=1, sim_options=sim_options, bsl_options=bsl_options)

# 4. 模型训练
algo.fit(trainset)  # 训练

# 5. 模型评估
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

#作业代码
#获取的是每个用户对应的K个推荐商品列表
K=10
rec_path = 'output/UserCF_REC.csv'
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
print("UserCF运行结束。。。。。。")
