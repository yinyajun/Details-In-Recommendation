#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/5 17:51
# @Author  : Yajun Yin
# @Note    :

from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.sql import HiveContext
from pyspark.mllib.evaluation import RankingMetrics


sc = SparkContext(appName="ALS-new")
hc = HiveContext(sc)

train_file = "/home/yinyajun/tmp/train.data/part-00000"
test_file = "/home/yinyajun/tmp/test.data/part-00000"

NDCG_AT = 10
ALPHA = 40
ITERS = 10
LAMBDA = 0.1
RANK = 48


def cal_ndcg(model, test_data, k):
    test = test_data.map(lambda p: (p[0], p[1]))
    ret = model.predictAll(test) \
        .map(lambda r: (r.user, (r.product, r.rating))) \
        .groupByKey() \
        .mapValues(lambda l: sorted(l, key=lambda x: x[1], reverse=True)) \
        .mapValues(lambda l: [x[0] for x in l])
    true = test_data.filter(lambda p: p[2] == 1.0).map(lambda r: (r[0], [r[1]]))
    predictionAndLabels = ret.join(true).map(lambda r: (r[1][0], list(r[1][1])))
    metrics = RankingMetrics(predictionAndLabels)
    return metrics.ndcgAt(k)


train_data = sc.textFile(train_file).map(lambda p: eval(p)).map(lambda p: (p[0], p[1], p[3]))
test_data = sc.textFile(test_file).map(lambda p: eval(p)).map(lambda p: (p[0], p[1], p[3]))
# train
model = ALS.trainImplicit(train_data, RANK, iterations=ITERS, lambda_=LAMBDA, alpha=ALPHA)
ndcg = cal_ndcg(model, test_data, NDCG_AT)
print('ndcg:', ndcg)
