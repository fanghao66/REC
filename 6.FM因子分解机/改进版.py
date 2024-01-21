# -*- coding: utf-8 -*-
"""
FM实现召回和排序
"""
import os
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# region 模型结构
from torch.utils.data import dataloader, dataset


class SparseEmbedding(nn.Module):
    def __init__(self, sparse_field_nums, embed_dim):
        super(SparseEmbedding, self).__init__()
        self.sparse_num_field = len(sparse_field_nums)  # 稀疏(离散的分类)特征的数量
        self.embed_layer = nn.Embedding(num_embeddings=sum(sparse_field_nums), embedding_dim=embed_dim)
        self.offsets = np.asarray((0, *np.cumsum(sparse_field_nums)[:-1]), dtype=np.int32)

    def forward(self, x):
        """
        前向过程
        :param x: 输入的原始系数特征id [N,sparse_num_field]
        :return: [N,sparse_num_field,embed_dim]
        """
        x = x + x.new_tensor(self.offsets)
        z = self.embed_layer(x)  # [N,sparse_num_field,embed_dim]
        return z


class DenseEmbedding(nn.Module):
    def __init__(self, dense_num_field, embed_dim):
        super(DenseEmbedding, self).__init__()
        self.dense_num_field = dense_num_field  # 稠密(连续)特征的数量
        self.dense_w = nn.Parameter(torch.empty(1, self.dense_num_field, embed_dim))
        nn.init.normal_(self.dense_w)

    def forward(self, x):
        """
        前向过程
        :param x: 原始输入的稠密特征向量x [N,dense_num_field]
        :return: [N,dense_num_field,embed_dim]
        """
        x = x.view(-1, self.dense_num_field, 1)  # [N,dense_num_field] -> [N,dense_num_field,1]
        # [1,dense_num_field,embed_dim] * [N,dense_num_field,1] -> [N,dense_num_field,embed_dim]
        z = self.dense_w * x
        return z


class SideFMVectorBaseModule(nn.Module):
    def __init__(self, sparse_field_nums, dense_num_field, embed_dim, is_spu_side=False):
        super(SideFMVectorBaseModule, self).__init__()
        self.is_spu_side = is_spu_side
        # 1阶部分
        self.sparse_linear = SparseEmbedding(sparse_field_nums, embed_dim=1)
        self.dense_linear = DenseEmbedding(dense_num_field, embed_dim=1)
        # 2阶部分
        self.sparse_second_order = SparseEmbedding(sparse_field_nums, embed_dim=embed_dim)
        self.dense_second_order = DenseEmbedding(dense_num_field, embed_dim=embed_dim)

    def internal_forward(self, sparse_x, dense_x):
        # 1阶部分
        # [N,sparse_num_field] -> [N,sparse_num_field,1] -> [N]
        v1_sparse = self.sparse_linear(sparse_x).squeeze(-1).sum(dim=1)
        # [N,dense_num_field] -> [N,dense_num_field,1] -> [N]
        v1_dense = self.dense_linear(dense_x).squeeze(-1).sum(dim=1)
        v1 = v1_sparse + v1_dense  # [N]

        # 2阶部分
        # [N,sparse_num_field] -> [N,sparse_num_field,embed_dim]
        v2_sparse = self.sparse_second_order(sparse_x)
        # [N,dense_num_field] -> [N,dense_num_field,embed_dim]
        v2_dense = self.dense_second_order(dense_x)
        # num_field = sparse_num_field + dense_num_field
        # 合并 [N,num_field,embed_dim]
        v2 = torch.concat([v2_sparse, v2_dense], dim=1)

        return v1, v2

    def forward(self, sparse_x, dense_x):
        return self.internal_forward(sparse_x, dense_x)

    def get_vectors(self, sparse_x, dense_x):
        """
        返回样本对应的向量 --> 可用于召回阶段
        :param sparse_x: [N,sparse_num_field] 稀疏特征
        :param dense_x: [N,dense_num_field] 稠密特征
        :return: [N,embed_dim+1]
        """
        # v1:[N];    针对每个样本存在一个一维的置信度值
        # v2:[N,num_field,embed_dim] 针对每个样本都存在num_field个的embed_dim维度的特征向量;
        # num_field=sparse_num_field+dense_num_field
        v1, v2 = self.internal_forward(sparse_x, dense_x)

        # 向量直接将所有field的向量累计即可
        v = v2.sum(dim=1)  # [N,num_field,embed_dim] -> [N,embed_dim]

        # 将1阶部分的值，加入到最终向量的尾部
        v1 = v1.view(-1, 1)  # [N] -> [N,1]
        if self.is_spu_side:
            print("当前是物品侧向量子模型....")
            # 商品侧自身和自身的二阶部分的置信度
            square_sum = v2.sum(dim=1).pow(2).sum(dim=1)
            # [N,num_field,embed_dim] -> [N,num_field,embed_dim] -> [N,embed_dim] -> [N]
            sum_square = v2.pow(2).sum(dim=1).sum(dim=1)
            v2_v1 = 0.5 * (square_sum - sum_square)  # [N]
            v2_v1 = v2_v1.view(-1, 1)
            v1 = v1 + v2_v1
        else:
            # 当前是用户侧子模型，直接填充1即可；所以重置v1=1
            v1 = torch.ones_like(v1)
        v = torch.concat([v, v1], dim=1)  # [N,embed_dim] concat [N,1] -> [N,embed_dim+1]

        return v


# noinspection DuplicatedCode
class FM(nn.Module):
    def __init__(self, user_sparse_field_nums, user_dense_num_field, spu_sparse_field_nums, spu_dense_num_field,
                 embed_dim):
        """
        :param user_sparse_field_nums: 所有用户稀疏特征对应的每个特征的类别数目，eg: [10000,3,100,50]
        :param user_dense_num_field: 稠密特征的数量
        :param spu_sparse_field_nums: 所有商品稀疏特征对应的每个特征的类别数目，eg: [5000,300]
        :param spu_dense_num_field: 稠密特征的数量
        :param embed_dim: 二阶特征部分，映射的向量维度大小
        """
        super(FM, self).__init__()
        self.register_buffer('user_sparse_field_nums', torch.tensor(user_sparse_field_nums))
        self.register_buffer('user_dense_num_field', torch.tensor(user_dense_num_field))
        self.register_buffer('spu_sparse_field_nums', torch.tensor(spu_sparse_field_nums))
        self.register_buffer('spu_dense_num_field', torch.tensor(spu_dense_num_field))
        self.register_buffer('embed_dim', torch.tensor(embed_dim))
        # 0阶部分
        self.bias = nn.Parameter(torch.zeros(1))
        # 用户侧子模型
        self.user_side = SideFMVectorBaseModule(
            user_sparse_field_nums, user_dense_num_field, embed_dim, is_spu_side=False
        )
        # 商品侧子模型
        self.spu_side = SideFMVectorBaseModule(
            spu_sparse_field_nums, spu_dense_num_field, embed_dim, is_spu_side=True
        )

    def forward(self, user_sparse_x, user_dense_x, spu_sparse_x, spu_dense_x):
        # 1. 提取用户侧的相关信息
        # v1_user:[N];    v2_user:[N,user_num_field,embed_dim]
        v1_user, v2_user = self.user_side(user_sparse_x, user_dense_x)
        # 2. 提取物品侧的相关信息
        # v1_spu:[N];    v2_spu:[N,spu_num_field,embed_dim]
        v1_spu, v2_spu = self.spu_side(spu_sparse_x, spu_dense_x)

        # 1阶部分
        v1 = v1_spu + v1_user

        # 2阶部分
        # 合并 [N,num_field,embed_dim]
        v2 = torch.concat([v2_user, v2_spu], dim=1)
        # 快速计算
        # [N,num_field,embed_dim] --> [N,embed_dim] --> [N,embed_dim] --> [N]
        square_sum = v2.sum(dim=1).pow(2).sum(dim=1)
        # [N,num_field,embed_dim] -> [N,num_field,embed_dim] -> [N,embed_dim] -> [N]
        sum_square = v2.pow(2).sum(dim=1).sum(dim=1)
        v2 = 0.5 * (square_sum - sum_square)

        # 合并0、1、2三个部分的置信度
        z = self.bias + v1 + v2
        return z


def t0():
    fmnet = FM(
        user_sparse_field_nums=[1000, 2000, 355, 140, 250],
        user_dense_num_field=10,
        spu_sparse_field_nums=[5222, 352, 1000],
        spu_dense_num_field=23,
        embed_dim=5
    )
    batch_size = 2
    user_sparse_x = torch.randint(100, (batch_size, 5))
    user_dense_x = torch.randn(batch_size, 10)
    spu_sparse_x = torch.randint(100, (batch_size, 3))
    spu_dense_x = torch.randn(batch_size, 23)
    r = fmnet(user_sparse_x, user_dense_x, spu_sparse_x, spu_dense_x)
    print(r)

    user_vector = fmnet.user_side.get_vectors(user_sparse_x, user_dense_x)
    print(user_vector)

    spu_vector = fmnet.spu_side.get_vectors(spu_sparse_x, spu_dense_x)
    print(spu_vector)
t0()