#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/12 16:55
# @Author  : Yajun Yin
# @Note    :


import tensorflow as  tf
import pandas as pd
import numpy as np
import os

DATA_FILE = "ratings.dat"


def split_data(filename):
    df = pd.read_csv(filename, sep='::', names=['user', 'item', 'label', 'time'])
    del df['time']
    # reindex
    user, user_index = np.unique(df.user, return_inverse=True)
    item, item_index = np.unique(df.item, return_inverse=True)
    user_size = len(user)
    item_size = len(item)
    data_size = len(df)
    df.user = user_index
    df.item = item_index
    print("user_size:", user_size)
    print("item_size", item_size)
    print("data_size:", data_size)
    shuffled_df = df.sample(frac=1)

    train_size = int(data_size * 0.95)
    validation_size = int(data_size * 0.05)

    train, val, test = np.split(shuffled_df, [train_size, validation_size + train_size])
    train.to_csv("./train.data", header=None, index=None)
    val.to_csv("./validation.data", header=None, index=None)
    # test.to_csv("./test.data", header=None, index=None)
    # os.remove(filename)


if __name__ == '__main__':
    split_data(DATA_FILE)
    print('data set split ok!')
