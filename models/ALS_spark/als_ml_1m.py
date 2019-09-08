#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/14 18:40
# @Author  : Yajun Yin
# @Note    :


from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.sql import HiveContext
from pyspark.mllib.evaluation import RankingMetrics
import time

sc = SparkContext(appName="ALS-ml-1m")
hc = HiveContext(sc)

train_file = "/home/hdp-huajiao/yinyajun/tmp/ml-1m-train.rating"
test_file = "/home/hdp-huajiao/yinyajun/tmp/ml-1m-test.rating"

TopK = 10
# ALPHA = 40.0
# ITERS = 10
# LAMBDA = 0.01
FACTOR = 48


def build_model(train_data):
    model = ALS.trainImplicit(train_data,
                              rank=FACTOR,
                              iterations=ITERS,
                              lambda_=LAMBDA,
                              alpha=ALPHA)
    return model


def evaluate(model, test_data):
    test = test_data.map(lambda p: (p[0], p[1]))
    ret = model.predictAll(test) \
        .map(lambda r: (r.user, (r.product, r.rating))) \
        .groupByKey() \
        .mapValues(lambda l: sorted(l, key=lambda x: x[1], reverse=True)) \
        .mapValues(lambda l: [x[0] for x in l])
    gt_items = test_data.filter(lambda p: p[2] == 1.0).map(lambda r: (r[0], [r[1]]))
    predictionAndLabels = ret.join(gt_items).map(lambda r: (r[1][0], list(r[1][1])))
    metrics = RankingMetrics(predictionAndLabels)
    return metrics.ndcgAt(TopK)


def main():
    train_data = sc.textFile(train_file).map(lambda p: eval(p)).map(lambda p: (p[0], p[1], p[2]))
    test_data = sc.textFile(test_file).map(lambda p: eval(p)).map(lambda p: (p[0], p[1], p[2]))
    model = build_model(train_data)
    ndcg = evaluate(model, test_data)
    print("ITER: {0}, LAMBDA: {1}, ALPHA: {2}, NDCG@{3}: {4}".format(ITERS, LAMBDA, ALPHA, TopK, ndcg))


if __name__ == '__main__':
    iters = [5, 10, 20, 40, 100]
    _lambda = [0.01, 0.05, 0.1, 0.5, 1]
    alphas = [1.0, 5.0, 20.0, 40.0, 120.0]
    for i in iters:
        for j in _lambda:
            for p in alphas:
                ITERS = i
                LAMBDA = j
                ALPHA = p
                main()
