#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/14 19:06
# @Author  : Yajun Yin
# @Note    :


import pandas as pd
import numpy as np
from tqdm import tqdm

TEST_SAM = 9
TRAIN_SUM = 4


class MovieLens1M(object):
    """
    Leave one out to split train and test.
    For each user, the recent interactive item is in test.
    Add another k un-interactive items for each user as negative examples.
    So, in test:
    u1: [gt_item, neg_1,..., neg_k]
    u2: [gt_item, neg_1,..., neg_k]
    ...
    """

    def __init__(self, file_dir='ratings.dat'):
        self.file_dir = file_dir
        self.user_count = 0
        self.item_count = 0
        self.train = None
        self.test = None
        self.user_interactions = None

        self.load_ml_1m()

    def load_ml_1m(self):
        df = pd.read_csv(self.file_dir, sep='::', names=["user", "item", "score", "time"])
        # reindex(start from zero)
        user, user_index = np.unique(df.user, return_inverse=True)
        item, item_index = np.unique(df.item, return_inverse=True)
        df.user = user_index
        df.item = item_index
        self.item_count = max(item_index) + 1
        self.user_count = max(user_index) + 1

        # sort by user, time
        df = df.sort_values(by=['user', 'time'])

        # split (train = df - test)
        self.test = df.groupby("user", as_index=False).last()
        tmp = df.merge(self.test, on=['user', 'item', 'score'], how="left")
        self.train = tmp.loc[tmp.time_x != tmp.time_y, :].iloc[:, :4]
        self.train.columns = ["user", "item", "score", "time"]

        self.user_interactions = df.groupby("user").apply(lambda p: np.array(p.item))

    def negative_sample(self, dataset, k):
        ret = pd.DataFrame(columns=["user", 'item', "score"])
        for i in tqdm(range(len(dataset))):
            row = dataset.iloc[i, :]
            user = row['user']
            item = row["item"]
            score = row['score']
            interactions = self.user_interactions[user]
            sample_pool = np.delete(np.arange(self.item_count), interactions)
            samples = np.random.choice(sample_pool, size=k)
            # add ground truth to ret
            groud_truth = pd.DataFrame({'user': [user], 'item': [item], 'score': [score]})
            ret = ret.append(groud_truth)
            # add negative to ret
            negative_part = pd.DataFrame({'user': [user] * k, 'item': samples, 'score': [0.0] * k})
            ret = ret.append(negative_part)
            # format
        ret = ret.loc[:, ["user", 'item', "score"]]
        return ret

    def ncf_dataset(self):
        test = self.negative_sample(self.test, TEST_SAM)
        train = self.negative_sample(self.train, TRAIN_SUM)
        return train, test

    def implicit_dataset(self):
        test = self.negative_sample(self.test, TEST_SAM)
        train = self.train.loc[:, ["user", 'item', "score"]]
        return train, test

    def fast_als_dataset(self):
        test = self.test.loc[:, ["user", 'item', "score"]]
        train = self.train.loc[:, ["user", 'item', "score"]]
        return train, test

    @staticmethod
    def save_dataset(dataset, file_dir):
        dataset.to_csv(file_dir, header=None, index=None)


if __name__ == '__main__':
    m = MovieLens1M()
    train, test = m.implicit_dataset()
    m.save_dataset(train, "ml-1m-train.rating")
    m.save_dataset(test, "ml-1m-test.rating")
