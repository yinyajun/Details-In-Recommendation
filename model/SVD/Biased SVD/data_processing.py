#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/4 16:24
# @Author  : Yajun Yin
# @Note    :

import pandas as pd
import numpy as np


def split_data(filename):
    """ split depend on timestamp"""
    columns = ['user', 'item', 'score', 'time']
    df = pd.read_csv(filename, sep='::', names=columns)
    data_size = len(df)
    print("data_size:", data_size)
    print("user_size:", len(df.user.unique()))  # 69878
    print("item_size:", len(df.item.unique()))  # 10677

    # reindex
    user, user_index = np.unique(df.user, return_inverse=True)
    item, item_index = np.unique(df.item, return_inverse=True)
    df.user = user_index
    df.item = item_index

    # split depend on time
    test = df[df.time > 1210000000]
    index = df.index
    test_index = test.index
    train_index = [i for i in index if i not in test_index]
    train = df.iloc[train_index, :]

    # filter train
    tmp1 = train.groupby('user').count()
    user_index = tmp1[tmp1.item >= 20].index
    train = train[train.user.isin(user_index)]
    item_index = train.item.unique()

    # filter test
    test = test[test.user.isin(user_index)]
    test = test[test.item.isin(item_index)]

    # save
    train.to_csv("./train.data", header=None, index=None)
    test.to_csv("./validation.data", header=None, index=None)


if __name__ == '__main__':
    split_data(filename='./ratings.dat')
    print('data set split ok!')
