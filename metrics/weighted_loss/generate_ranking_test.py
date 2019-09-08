#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/24 18:50
# @Author  : Yajun Yin
# @Note    :
from __future__ import print_function
from pyspark import SparkConf, SparkContext

TEST_DATA = "hdfs://~/20180820/validation"
RANKING_DATA = "hdfs://~/20180820/ranking"

# TEST_DATA is csv formatted data
# e.g. a example like this "label,weight,uid,...,other features,..."

def contain_positive_label(features):
    for feature in features:
        label = feature.split(',')[0]
        if label == b'1' or label == '1':
            return True
    return False


conf = SparkConf().setAppName('get ranking data')
sc = SparkContext(conf=conf)

rdd = sc.textFile(TEST_DATA)
rdd1 = rdd.map(lambda p: (p.split(',')[2], p)) # key is uid
# per user has at least 10 examples in which at least 1 positive example exists.
rdd2 = rdd1.groupByKey().map(lambda p: (p[0], tuple(p[1]))) \
    .filter(lambda p: contain_positive_label(p[1])) \ # at least 1 positive example
    .filter(lambda p: len(p[1]) >= 10)
print("how many user:", rdd2.count())
rdd3 = rdd2.flatMap(lambda p: p[1])
print("how many examples:", rdd3.count())
rdd3.saveAsTextFile(RANKING_DATA)

