# -*- coding: utf-8 -*-
import os

from surprise import accuracy, Dataset, NormalPredictor, dump
from surprise.model_selection import train_test_split

# 1. 加载数据
data = Dataset.load_builtin("ml-100k")

# 2. 数据转换
# sample random trainset and testset 数据分割成75%的训练数据集 + 25%的测试数据集
# 训练数据的转换：[外部id转内部id的操作（用户内部字典，商品内部字典）、每个用户的评论商品列表、每个商品评论的用户列表、用户数量、商品数量、评论条数]。
# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=0.25)

# 3. 构建模型对象
algo = NormalPredictor()

# 4. 模型训练
# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)  # 训练

# 5. 模型评估
# Then compute RMSE
predictions = algo.test(testset)  # 预测，将rid和uid转换成内部字典的编码再进行预测，返回Pedictions类
accuracy.rmse(predictions)  # 评估
accuracy.mse(predictions)  # 评估
accuracy.fcp(predictions)  # 评估

# 6. 模型持久化
output_dir = "output/normal"
os.makedirs(output_dir, exist_ok=True)
dump.dump(f"{output_dir}/model.pkl", predictions=None, algo=algo)
_, algo = dump.load(f"{output_dir}/model.pkl")


# 7. 模型的预测
print("=" * 100)
#预测和评估的区别就是一个处理批量数据，一个处理单个数据
y_ = algo.predict(uid="196", iid="224", r_ui=None, clip=True)
print(y_)
print(f"预测评分:{y_.est:.2f}")
y_ = algo.predict(uid="196", iid="224", r_ui=3.5, clip=True)
print(y_)
print(f"预测评分:{y_.est:.2f}")
y_ = algo.predict(uid="196", iid="224", r_ui=3.5, clip=True)
print(y_)
print(f"预测评分:{y_.est:.2f}")
