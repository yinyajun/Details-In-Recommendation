#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/12 16:55
# @Author  : Yajun Yin
# @Note    :


import tensorflow as  tf
import csv
import pandas as pd
import numpy as np
import os


def preprocessing():
    with open("./ratings.csv") as fr:
        with open("./data", 'w', newline='') as fw:
            reader = csv.reader(fr)
            writer = csv.writer(fw)
            for row in reader:
                writer.writerow(row[:3])
    print("data rewrite finish")


def split_data(filename):
    df = pd.read_csv(filename)
    data_size = len(df)
    print("data_size:", data_size)
    shuffled_df = df.sample(frac=1)

    train_size = int(data_size * 0.7)
    validation_size = int(data_size * 0.1)

    train, val, test = np.split(shuffled_df, [train_size, validation_size + train_size])
    train.to_csv("./train.data", header=None, index=None)
    val.to_csv("./validation.data", header=None, index=None)
    test.to_csv("./test.data", header=None, index=None)
    os.remove(filename)


if __name__ == '__main__':
    preprocessing()
    split_data("./data")
