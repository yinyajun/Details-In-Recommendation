#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/19 15:54
# @Author  : Yajun Yin
# @Note    :


from __future__ import print_function

from pyspark import SparkConf, SparkContext
from pyspark.sql import HiveContext
from pyspark.sql import functions as F
from pyspark.sql import Window
from random import sample


class EmptyDataFrameError(Exception):
    def __init__(self, err="dataframe is empty!"):
        Exception.__init__(self, err)


def _split_observed_pair(observed_pair):
    """
    split positive examples to _train and _test
    _test: positive examples of test dataset
    _train: positive examples of train dataset
    """
    df = observed_pair
    df.cache()
    # get recent consumed item
    window = Window.partitionBy('user').orderBy('time').rowsBetween(-10000, 10000)  # default range is (-infty, 0)
    last_df = df.select('user', 'item', 'score', 'time', F.last('time').over(window).alias('last_time')) \
        .filter('time=last_time')  # in the same time more than 1 item consumed
    last_df = last_df.dropDuplicates(['user', 'time'])  # ensure only 1 item for each user at each time
    # get test and train
    test = last_df.select(['user', 'item', 'score'])
    train = df.select('user', 'item', 'score').subtract(test)
    df.unpersist()
    return train, test


def _get_neg(observed_df, **kwargs):
    observed_df = observed_df.select('user', 'item', 'score')
    # in spark1.6, collect_list is not supported in a window operation
    # window = Window.partitionBy('user')
    # df = observed_df.withColumn("observed_list", F.collect_list('item').over(window))
    observed_df.registerTempTable('tmp')
    sql = """
    select 
        user, item, score, collect_list(item) over(partition by user) as observed_list
    from 
        tmp
    """
    df = hc.sql(sql)
    item_size = kwargs['item_size']

    def _get_unobserved_item_list(observed_list):
        return [i for i in range(item_size) if i not in observed_list]

    df = df.map(list).map(lambda p: (p[0], p[1], p[2], _get_unobserved_item_list(p[3]))) \
        .toDF(['user', 'item', 'score', 'item_list'])
    return df


class ImplicitDataGenerator(object):
    """
    Construct dataset for implicit feedback.

    For implicit feedback (e.g. watch time, clicks, purchases),
    any observed user-item-score example will be labeled 1. (score > 0.0)
    To the contrary, any unobserved example will be labeled 0. (score == 0.0)

    Raw dataset contains m observed user-item-score examples. Assume user size is k and item size is v.

    * implicit feedback dataset size is k*v, m of which are labeled 1 and the others are labeled 0.
    * NCF dataset need further processing. Not all unobserved examples are included, NCF dataset use
    negative sampling. For each user, only limited unobserved items are included.

    Require raw dataset:
    * COO format (user-item-score)
    * Ensure that each user consumes at least 1 item
    * Ensure that each item is consumed by at least 1 user
    """

    def __init__(self, raw_df, min_user_num=0, min_item_num=0, train_neg_num=2, test_neg_num=9):
        self.raw_df = raw_df
        self.user_size = None
        self.item_size = None
        self.min_user_num = min_user_num
        self.min_item_num = min_item_num
        self._observed_pair = None
        self.train_neg_num = train_neg_num
        self.test_neg_num = test_neg_num
        self.train_df = None
        self.test_df = None
        #
        self.process()

    @staticmethod
    def _filter_and_check(df, min_user_num, min_item_num):
        """
        The performance of CF for cold start is not good.
        If a user seldom consume items or a item seldom consumed by users, it will lead to bad performance.
        ensure min_item_num and min_user_num are not very small, respectively.
        """
        # min_item and min_user
        window1 = Window.partitionBy("user")
        window2 = Window.partitionBy("item")
        df1 = df.withColumn("item_num", F.count("user").over(window1)) \
            .withColumn("user_num", F.count("item").over(window2))
        df2 = df1.filter("item_num>=%s" % str(min_item_num)) \
            .filter("user_num>=%s" % str(min_user_num))
        data_size = df2.count()
        if data_size == 0:
            raise EmptyDataFrameError()
        else:
            print("After filter low_freq user and low_freq item, data size is %d" % data_size)
        return df2.drop('user_num').drop('item_num')

    def _reindex(self, df):
        """
        # distinct_user = df2.select("user").distinct()
        # distinct_item = df2.select("item").distinct()
        # window = Window.partitionBy()  # window is whole df
        # user_index = distinct_user.select("user", (F.row_number().over(window) - 1).alias('user_index'))
        # item_index = distinct_item.select("item", (F.row_number().over(window) - 1).alias("item_index"))
        # user_index_map = sc.broadcast(user_index.rdd.map(tuple).collectAsMap())
        # item_index_map = sc.broadcast(item_index.rdd.map(tuple).collectAsMap())
        """
        user_list = df.select(F.collect_set("user"))
        item_list = df.select(F.collect_set("item"))
        user_index_map = sc.broadcast(user_list.flatMap(lambda p: zip(p[0], range(len(p[0])))).collectAsMap())
        item_index_map = sc.broadcast(item_list.flatMap(lambda p: zip(p[0], range(len(p[0])))).collectAsMap())

        def _sub(example):
            user = example[0]
            example[0] = user_index_map.value[user]
            item = example[1]
            example[1] = item_index_map.value[item]
            return example

        self.user_index_map = user_index_map
        self.item_index_map = item_index_map
        self.user_size = len(self.user_index_map.value)
        self.item_size = len(self.item_index_map.value)
        print("user size is %d" % self.user_size)
        print("item size is %d" % self.item_size)
        columns = df.columns
        return df.map(list).map(lambda p: _sub(p)).toDF(columns)

    def process(self):
        """
        user, item, score, time, label
        """
        # check and reindex
        df = self._filter_and_check(self.raw_df, self.min_user_num, self.min_item_num)
        df = self._reindex(df)
        self._observed_pair = df
        return df

    @staticmethod
    def _negative_sample(dataset, neg_num):
        """data
        dataset like user, item, score, items_list
        """
        columns = dataset.columns
        assert 'user' in columns and 'item' in columns and 'score' in columns and 'item_list' in columns
        df = dataset.select('user', 'item', 'score', 'item_list')

        def _add_score(sample_items, pos_item, score):
            assert len(sample_items) == neg_num
            all_item_list = [pos_item] + sample_items
            score_list = [0.0] * (neg_num + 1)
            score_list[0] = score
            return list(zip(all_item_list, score_list))

        df = df.map(lambda p: (p[0], p[1], p[2], sample(p[3], neg_num))) \
            .map(lambda p: (p[0], _add_score(p[3], p[1], p[2]))) \
            .flatMapValues(lambda p: p) \
            .map(lambda p: (p[0], p[1][0], p[1][1])) \
            .toDF(['user', 'item', 'score'])
        return df

    def custom_generate_NCF_dataset(self,
                                    pos_split_fn=_split_observed_pair,
                                    get_train_neg=_get_neg,
                                    get_test_neg=_get_neg):
        """fsdafs
        a = NCF()
        df = a.dataset
        train = df.train_df


        """
        observed_pair = self._observed_pair if self._observed_pair else self.process()
        train_pos, test_pos = pos_split_fn(observed_pair)
        train = get_train_neg(train_pos, item_size=self.item_size)
        test = get_test_neg(test_pos, item_size=self.item_size)
        #
        train_df = self._negative_sample(train, self.train_neg_num)
        self.train_df = train_df.withColumn('label', F.when(F.col('score') > 0.0, 1.0).otherwise(0.0))
        test_df = self._negative_sample(test, self.test_neg_num)
        # add label for test
        self.test_df = test_df.withColumn('label', F.when(F.col('score') > 0.0, 1.0).otherwise(0.0))
        return self

    def _get_all_pair(self, observed_df):
        observed_df = observed_df.select('user', 'item', 'score')
        df = observed_df.groupBy('user').agg(F.collect_list('item').alias("observed_list"),
                                             F.collect_list('score').alias('score_list'))
        item_size = self.item_size

        def _treat(observed_list, score_list):
            unobserved_list = [i for i in range(item_size) if i not in observed_list]
            all_item_list = observed_list + unobserved_list
            all_score_list = score_list + [0.0] * len(unobserved_list)
            return list(zip(all_item_list, all_score_list))

        df1 = df.map(tuple).map(lambda p: (p[0], _treat(p[1], p[2]))).flatMapValues(lambda p: p) \
            .map(lambda p: (p[0], p[1][0], p[1][1])).toDF(['user', 'item', 'score'])
        return df1

    def generate_implicit_dataset(self):
        """['user', 'item', 'score', 'item_list']"""
        observed_pair = self._observed_pair if self._observed_pair else self.process()
        train_pos, test_pos = _split_observed_pair(observed_pair)
        test = _get_neg(test_pos, item_size=self.item_size)

        def _add_score(unobserved_items, pos_item, score):
            all_item_list = [pos_item] + unobserved_items
            score_list = [0.0] * len(all_item_list)
            score_list[0] = score
            return list(zip(all_item_list, score_list))

        test_df = self._negative_sample(test, self.test_neg_num)
        self.test_df = test_df.withColumn('label', F.when(F.col('score') > 0.0, 1.0).otherwise(0.0))
        train_df = self._get_all_pair(train_pos)
        self.train_df = train_df.withColumn('label', F.when(F.col('score') > 0.0, 1.0).otherwise(0.0))
        return self

    def generate_NCF_dataset(self, pos_split_fn=_split_observed_pair):
        observed_pair = self._observed_pair if self._observed_pair else self.process()
        train_pos, test_pos = pos_split_fn(observed_pair)
        train_df = self._add_sampled_pair(train_pos, self.train_neg_num)
        self.train_df = train_df.withColumn('label', F.when(F.col('score') > 0.0, 1.0).otherwise(0.0))
        test_df = self._add_sampled_pair(test_pos, self.test_neg_num)
        self.test_df = test_df.withColumn('label', F.when(F.col('score') > 0.0, 1.0).otherwise(0.0))
        return self

    def _add_sampled_pair(self, observed_df, neg_num):
        observed_df = observed_df.select('user', 'item', 'score')
        df = observed_df.groupBy('user').agg(F.collect_list('item').alias("observed_list"),
                                             F.collect_list('score').alias('score_list'))
        item_size = self.item_size

        def _get_neg_and_sample(observed_list, score_list, neg_num):
            unobserved_list = [i for i in range(item_size) if i not in observed_list]
            tmp = []
            for i in range(len(observed_list)):
                a = observed_list[i]
                b = sample(unobserved_list, neg_num)
                c = [a] + b
                d = [0.0] * (neg_num + 1)
                d[0] = score_list[i]
                for j in list(zip(c, d)):
                    tmp.append(j)
            return tmp

        df1 = df.map(tuple).map(lambda p: (p[0], _get_neg_and_sample(p[1], p[2], neg_num))) \
            .flatMapValues(lambda p: p) \
            .map(lambda p: (p[0], p[1][0], p[1][1])) \
            .toDF(['user', 'item', 'score'])
        return df1

    @staticmethod
    def save_dataset(dataset, target_dir, shuffle=False):
        if shuffle:
            dataset = dataset.sample(False, 1.0)
        rdd = dataset.map(tuple).map(lambda p: tuple(map(str, p))).map(lambda p: ','.join(p))
        rdd.coalesce(1).saveAsTextFile(target_dir)


if __name__ == '__main__':
    raw_file = "/home/company/yinyajun/tmp/ratings.csv"
    train_file = "/home/company/yinyajun/tmp/train_mv_1"
    test_file = "/home/company/yinyajun/tmp/test_mv_1"

    # init spark context
    conf = SparkConf().setAppName("mf dataset")
    conf.set("spark.hadoop.validateOutputSpecs", "false")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.speculation", "true")
    sc = SparkContext(conf=conf)
    hc = HiveContext(sc)

    rdd = sc.textFile(raw_file)
    rdd = rdd.map(lambda row: row.strip().split(',')).map(
        lambda row: [int(row[0]), int(row[1]), float(row[2]), int(row[3])])
    raw_df = rdd.toDF(['user', 'item', 'score', 'time'])

    idg = ImplicitDataGenerator(raw_df, min_item_num=5, min_user_num=5, train_neg_num=4, test_neg_num=99)
    idg.generate_implicit_dataset()
    idg.save_dataset(idg.train_df, train_file, shuffle=True)
    idg.save_dataset(idg.test_df, test_file)

