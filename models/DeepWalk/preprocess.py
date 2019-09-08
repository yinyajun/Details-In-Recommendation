#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/2 19:05
# @Author  : Yajun Yin
# @Note    :

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def session_segmentation(data, session_duration):
    # data: [(t1, item), (t2, item), (t3, item)]
    tmp = []
    session_id = 0
    for i in range(len(data)):
        if i > 0:
            difference = data[i][0] - data[i - 1][0]
            if difference > session_duration:  # 根据时间差来划分session
                session_id += 1
            elif data[i][1] == data[i - 1][1]:  # 如果同一个session内连续交互同一个item，视为一次
                continue
        tmp.append((session_id, data[i][1]))
    return tmp


def session_aggregation(data):
    # data: [(session_id, item), (session_id, item)...]
    # [(0, 2), (0, 3), (1, 5)] => [[2,3], [5]]
    ret = []
    sess = []
    for i in range(len(data)):
        if i > 0:
            if data[i][0] != data[i - 1][0]:
                ret.append(sess)
                sess = []
        sess.append(data[i][1])
    ret.append(sess)  # 将最后一次的sess放入ret
    ret = filter(lambda p: len(p) > 1, ret)  # 过滤掉session中只有一个item的session，它们不能提供图的边
    return ret


def adjacency_pair(data):
    # data: [item1,item2,item3] = [(item1, item2), (item2, item3)]
    ret = []
    for i in range(len(data)):
        if i > 0:
            pair = (data[i - 1], data[i])
            ret.append(pair)
    return ret


def adjacency_weight(data):
    # data: [e1,e1,e2,e3] => [(e1,e2,e3), (2,1,1)]
    # counter is not supported in python2.6
    counter = {}
    for e in data:
        counter[e] = counter.get(e, 0) + 1
    nodes, weights = zip(*counter.items())
    return nodes, weights


def preprocessing(rdd, max_active_num, session_duration):
    """
    rdd: user, item, timestamp
    """
    rdd = rdd.map(lambda p: (p[0], (p[2], p[1])))  # (user, (time, item))
    rdd = rdd.groupByKey() \
        .filter(lambda p: 2 <= len(p[1]) < max_active_num) \
        .mapValues(lambda p: sorted(p, key=lambda k: k[0])) \
        .mapValues(lambda p: session_segmentation(p, session_duration))
    rdd = rdd.flatMapValues(session_aggregation) \
        .flatMapValues(adjacency_pair) \
        .map(lambda p: p[1]) \
        .filter(lambda p: p[0] != p[1])  # 包含所有边的rdd，包含重边,除去自身的环
    rdd = rdd.groupByKey().mapValues(adjacency_weight)  # 计算边的weight
    return rdd
