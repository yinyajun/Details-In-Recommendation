#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/3 11:22
# @Author  : Yajun Yin
# @Note    :

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import heapq
import random
import time
from collections import defaultdict, Iterable


class Graph(defaultdict):
    def __init__(self):
        super(Graph, self).__init__(lambda: [[], None])  # pickle not support lambda function

    def from_map(self, map):
        for k, v in map.items():
            self[k] = v
        return self

    def nodes(self):
        return self.keys()

    def degree(self, nodes=None):
        # 节点的出度
        if isinstance(nodes, Iterable):
            return dict((v, len(self[v][0])) for v in nodes)
        return len(self[nodes][0])

    def order(self):
        # 图的阶，就是顶点个数
        return len(self)

    def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
        """
        带重启的带权随机游走
        :param path_length: 随机游走长度
        :param alpha: 重启概率
        :param rand: random generator实例
        :param start: 随机游走的起始点
        :return:
        """
        G = self
        # 设定起点
        if start:
            path = [start]
        else:
            # 如果start没设定，从所有顶点中随机挑选
            path = [rand.choice(list(G.nodes()))]
        while len(path) < path_length:
            cur = path[-1]
            neighbour_nodes = G[cur][0]
            neighbour_weights = G[cur][1]  # weights=None即均匀采样
            if len(neighbour_nodes) > 0:
                # 游走到相邻节点
                if rand.random() >= alpha:
                    path.extend(choices(rand, neighbour_nodes, weights=neighbour_weights))
                    # 回到起始点
                else:
                    path.append(path[0])
            else:
                break
        return [str(node) for node in path]


def load_adjacency_list_from_spark(rdd):
    """
    从rdd中加载邻边数据，以构成图
    """
    G = Graph()
    data = rdd.collectAsMap()
    for key, value in data.items():
        start = key
        neighbours, weights = value
        G[start] = [neighbours, weights]
    return G


def calc_indegree(G):
    d = {}
    for v in G.values():
        ods = v[0]
        for i in ods:
            d[i] = d.get(i, 0) + 1
    nodes = [(i[1], i[0]) for i in d.items()]
    heapq.heapify(nodes)
    return heapq.nlargest(100, nodes), heapq.nsmallest(100, nodes)


def calc_outdegree(G):
    d = G.degree(nodes=G.nodes())
    nodes = [(i[1], i[0]) for i in d.items()]
    heapq.heapify(nodes)
    return heapq.nlargest(100, nodes), heapq.nsmallest(100, nodes)


# ============== helper ====================
# Pyspark environment in our company is Python2.6, and some useful functions are
# not supported in python2.6. Rewrite these functions to replace.
# PARAMETER CHECKING is omitted.

def accumulator(population):
    s = 0
    for i in population:
        s += i
        yield s


def choices(rand, population, weights=None, k=1):
    # copy from random.choices()
    from bisect import bisect
    if weights is None:
        total = len(population)
        return [population[int(rand.random() * total)] for i in range(k)]
    cum_weights = list(accumulator(weights))
    if len(cum_weights) != len(population):
        raise ValueError('The number of weights does not match the population')
    total = cum_weights[-1]
    return [population[bisect(cum_weights, rand.random() * total)] for i in range(k)]
