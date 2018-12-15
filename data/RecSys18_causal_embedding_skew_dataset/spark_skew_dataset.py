#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import datetime

from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.sql import HiveContext
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import functions as F
from pyspark.sql import Window
import numpy as np
import time
import random
import math


class SkewDataset(object):
    def __init__(self, ratings, test_ratings, threshold=30.0):
        self.ratings = ratings
        self.test_ratings = test_ratings
        self.threshold = threshold
        self.s_t = None
        self.s_c = None
        self.test = None
        self.user_index_map = None
        self.item_index_map = None
        self._reindex()
        self._load()
        self.get_uniform_dataset()
        self.reindex_control()
    def _reindex(self):
        df = self.ratings
        if not self.user_index_map and not self.item_index_map:
            user_list = df.select(F.collect_set("user"))
            item_list = df.select(F.collect_set("item"))
            self.user_index_map = sc.broadcast(user_list.flatMap(lambda p: zip(p[0], range(len(p[0])))).collectAsMap())
            self.item_index_map = sc.broadcast(item_list.flatMap(lambda p: zip(p[0], range(len(p[0])))).collectAsMap())
        user_index_map = self.user_index_map
        item_index_map = self.item_index_map
        def _sub(example):
            user = example[0]
            example[0] = user_index_map.value[user]
            item = example[1]
            example[1] = item_index_map.value[item]
            return example
        self.user_size = len(self.user_index_map.value)
        self.item_size = len(self.item_index_map.value)
        print("user size is %d" % self.user_size)
        print("item size is %d" % self.item_size)
        columns = df.columns
        self.ratings = df.map(list).map(lambda p: _sub(p)).toDF(columns)
    def _set_label(self, data):
        return data.ratings.withColumn('score', F.when(F.col('score') >= self.threshold, 1.0).otherwise(0.0))
    def _load(self):
        # binarize the rating
        data = self._set_label(self.ratings)
        # least popular item
        item_counter = data.groupBy('item').count()
        min_count = item_counter.select(F.min('count')).first()[0]
        # accept probability
        acc_prob = item_counter.withColumn('acc', min_count / F.col('count'))
        # join accept prob
        df = data.join(acc_prob, 'item')
        self.data = df
        df.cache()
        print("===== Data load finished =====")
        count_result = df.agg(F.countDistinct('item'), F.countDistinct('user'), F.count('score')).first()
        self.num_user = count_result[1]
        self.num_item = count_result[0]
        self.data_num = count_result[2]
        print("Dataset size: ", self.data_num)
        print("User size: ", self.num_user)
        print("Item size: ", self.num_item)
    def get_uniform_dataset(self):
        """For the skewed data, first generate uniform dataset s_t using accept probability"""
        uniform_dataset = self.data.select('user', 'item', 'score', 'acc').map(list) \
            .map(lambda p: (p[0], p[1], p[2], np.random.binomial(1, p[3]))) \
            .toDF(['user', 'item', 'score', 'is_acc']) \
            .filter('is_acc=1').drop('is_acc')
        self.s_t = uniform_dataset
        # Taking uniform dataset s_t away, rest dataset is s_c
        data = self.data.select('user', 'item', 'score')
        self.s_c = self.data.subtract(uniform_dataset)
    def reindex_control(self):
        self.s_c = self.s_c.withColumn('item', F.col('item') + self.num_item)
    def _test(self):
        # filter new user and new item
        test = self.test
        test = test[test.isin(self.user_index_map.value.keys())]
        test = test[test.isin(self.item_index_map.value.keys())]
        test = self._set_label(test)
        positive = test[test.score == 1.0]
        window = Window().partitionBy(['user', 'item'])
        tmp = positive.select('user', F.last('item').over(window).alias('last_item'), 'score')
        df = tmp.join(positive, 'user')
        def _sample_pool(item_size, remove_list):
            sample_pool = list(range(item_size))
            for i in remove_list:
                sample_pool.remove(i)
            return sample_pool
        def _add_label(pos, negs):
            ret = []
            ret.append((pos, 1.0))
            for i in negs:
                ret.append((i, 0.0))
            return ret
        df = df.groupBy('user', 'last_item').agg(F.collect_list('item').alias('remove_list')) \
            .map(lambda p: (p[0], p[1], _sample_pool(self.num_item, p[2]))) \
            .map(lambda p: (p[0], p[1], np.random.choice(p[2], size=99, replace=False).tolist())) \
            .map(lambda p: (p[0], _add_label(p[1], p[2]))) \
            .flatMap(lambda p: p) \
            .map(lambda p: (p[0], p[1][0], p[1][1])) \
            .toDF(['user', 'item', 'score'])
        self.test = df


