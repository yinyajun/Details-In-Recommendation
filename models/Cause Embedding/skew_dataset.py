#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/27 18:14
# @Author  : Yajun Yin
# @Note    :

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd


class SkewDataset(object):
    def __init__(self, data_location, split_char=',', seed=123):
        self.data_location = data_location
        self.split_char = split_char
        self.seed = seed
        self._load()
        self.get_uniform_dataset()
        self.reindex_control()

    def _load(self):
        """load source data(MovieLens1M) and calculate accept probability for sampling"""
        df = pd.read_csv(self.data_location, sep=self.split_char, names=["user", "item", "rate", "time"])
        df = df.drop(columns=['time'])

        # use user_index to replace user, so does item
        user, user_index = np.unique(df.user, return_inverse=True)
        item, item_index = np.unique(df.item, return_inverse=True)
        df.user = user_index
        df.item = item_index

        # binarize the rating
        df.rate = df.rate.apply(lambda x: 1 if x >= 5.0 else 0)

        # least popular item
        counter = df['item'].value_counts()
        min_count = min(counter)
        assert min_count > 0

        # accept probability
        acc_prob = counter.apply(lambda p: 0.9 * min_count / p)  # cap the maximum probability to 0.9

        # transfer acc_prob(Series) to DataFrame
        acc_prob.name = "acc_prob"  # counter original name is 'item'
        acc_prob.index.name = "item"  # as a key to join
        df2 = acc_prob.reset_index()

        # merge df2 to  add column
        df = pd.merge(df, df2, on='item')
        self.df = df
        print("===== Data load finished =====")
        self.num_user = len(user)
        self.num_item = len(item)
        self.data_num = len(df)
        print("Dataset size: ", self.data_num)
        print("User size: ", self.num_user)
        print("Item size: ", self.num_item)

    def get_uniform_dataset(self):
        """For the skewed data, first generate uniform dataset s_t using accept probability"""
        uniform_size = int(np.ceil(0.3 * self.data_num))  # np.ceil return a float
        uniform_dataset = self.df.sample(uniform_size, weights=self.df.acc_prob, random_state=self.seed)
        self.s_t = uniform_dataset

        # Taking uniform dataset s_t away, rest dataset is s_c
        df_index = self.df.index
        s_t_index = self.s_t.index
        res_index = [i for i in df_index if i not in s_t_index]
        self.s_c = self.df.loc[res_index, :]

    def reindex_control(self):
        self.s_c.item = self.s_c.item + self.num_item

    def split(self):
        """split data to get train, validation and test
           - train : 0.6 s_c, 0.1 s_t
           - validation: 0.1 s_c
           - test: 0.2 s_t
        """
        # s_t: 0.3; s_c: 0.7
        s_t_num = len(self.s_t)
        s_c_num = len(self.s_c)
        test_num = int(np.ceil(s_t_num * 2 / 3))
        validation_num = int(np.ceil(s_c_num / 7))

        # Split
        test, train_st = np.split(self.s_t.sample(frac=1, random_state=self.seed), [test_num])
        validation, train_sc = np.split(self.s_c.sample(frac=1, random_state=self.seed), [validation_num])
        train = train_sc.append(train_st)

        # train set info
        train_num_user = len(train.user.unique())
        train_num_item = len(train.item.unique())

        # another test that all item occurs in train
        train_item = train.item.unique()
        new_test = test[test.item.isin(train_item)]

        # Print Schema info
        print("===== Begin split dataset =====")
        print("train size:", len(train))
        print("validation size:", len(validation))
        print("test size:", len(test))
        print("num of user in train:", train_num_user)
        print("num of item in train:", train_num_item)
        print("===== Split finished =====")

        columns = ['user', 'item', 'rate']
        train = train.loc[:, columns]
        test = test.loc[:, columns]
        validation = validation.loc[:, columns]
        new_test = new_test.loc[:, columns]
        return train, test, validation, new_test


def _sample_pool(item_size, remove_list):
    sample_pool = list(range(item_size))
    for i in remove_list:
        sample_pool.remove(i)
    return sample_pool


def evaluate_test(test_df, item_size):
    df = test_df
    tmp = df[df.rate > 0].groupby('user', as_index=False).last()
    df1 = pd.merge(tmp, df[df.rate > 0], on='user')
    sr = df1.groupby(['user', 'item_x']).item_y \
        .apply(list) \
        .apply(lambda p: _sample_pool(item_size, p)) \
        .apply(lambda p: np.random.choice(p, size=99, replace=False).tolist())
    # series to dataframe and set index as column
    df2 = sr.to_frame().reset_index(level=['user', 'item_x'])
    df2.columns = ['user', 'item', 'sample_list']
    data_df = pd.DataFrame(columns=['user', 'item', 'label'])
    for i in range(len(df2)):
        user = df2['user'][i]
        item = df2['item'][i]
        sample_list = df2['sample_list'][i]
        data_df = data_df.append(pd.DataFrame({'user': [user], 'item': [item], 'label': [1.0]}))
        record = {'user': [user] * 99, 'item': sample_list, 'label': [0.0] * 99}
        data_df = data_df.append(pd.DataFrame(record))
        data_df = data_df.loc[:, ["user", 'item', "label"]]
    return data_df


if __name__ == '__main__':
    data_file = "ratings.dat"
    df = SkewDataset(data_file)
    train, test, val, new_test = df.split()
    train.to_csv("user_prod_dict.skew.train.adapt_2i.csv", index=None, header=None)
    test.to_csv("user_prod_dict.skew.test.adapt_2i.csv", index=None, header=None)
    val.to_csv("user_prod_dict.skew.valid.adapt_2i.csv", index=None, header=None)
    new_test.to_csv("user_prod_dict.skew.new_test.adapt_2i.csv", index=None, header=None) # not yet use

    evaluate = evaluate_test(test, item_size=df.num_item)
    evaluate.to_csv("evaluate.csv", index=None, header=None)
