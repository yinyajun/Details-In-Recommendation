#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/16 16:14
# @Author  : Yajun Yin
# @Note    :

import pandas as pd
import numpy as np
from tqdm import tqdm


"""
download test from HDFS://home/hdp-huajiao/yinyajun/tmp/test_mv_1
"""

# read data
DATA_DIR = "./ratings.dat"
df = pd.read_csv(DATA_DIR, names=['user', 'item', 'label', 'time'])
del df['time']


# train and test
df['original_index'] = df.index
test_df = df.groupby('user', as_index=False).last()
res_index = [i for i in df.index if i not in test_df['original_index'].tolist()]
train_df = df.loc[df['original_index'].isin(res_index)].reset_index(drop=True)
del train_df['original_index']
del test_df['original_index']

# add negative sample
user_pos = {}
for i in range(len(df)):
    u = df['user'][i]
    t = df['item'][i]
    if u not in user_pos:
        user_pos[u] = []
    user_pos[u].append(t)

item_set = set(df.item.unique())

user_neg = {}
for key in user_pos:
    user_neg[key] = list(item_set - set(user_pos[key]))


def add_negative_sample(df, negative_num):
    data_df = pd.DataFrame(columns=["user", 'item', "label"])
    for i in tqdm(range(len(df))):
        user = df['user'][i]
        item = df['item'][i]
        # why add []? If using all scalar values, you must pass an index
        data_df = data_df.append(pd.DataFrame({'user': [user], 'item': [item], 'label': [1.0]}))
        negative_samples = np.random.choice(user_neg[user], size=negative_num, replace=False).tolist()
        record = {'user': [user] * negative_num, 'item': negative_samples, 'label': [0.0] * negative_num}
        data_df = data_df.append(pd.DataFrame(record))
        data_df = data_df.loc[:, ["user", 'item', "label"]]
    return data_df


# test
neg_num = 99
test_data_df = add_negative_sample(test_df, negative_num=neg_num)
test_data_df.to_csv("./evaluate.data", header=None, index=None)
print("save test ok!")

# train
neg_num = 4
train_data_df = add_negative_sample(train_df, negative_num=neg_num)
train_data_df = train_data_df.sample(frac=1)
train_data_size = len(train_data_df)
train_size = int(train_data_size * 0.9)

train, test = np.split(train_data_df, [train_size])

train.to_csv("./ncf_train.data", header=None, index=None)
test.to_csv("./ncf_validation.data", header=None, index=None)
print("save train ok!")
