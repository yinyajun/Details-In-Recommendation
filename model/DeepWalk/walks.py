#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/3 12:59
# @Author  : Yajun Yin
# @Note    :

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random
from .graph import Graph
from six.moves import zip_longest


def grouper(n, iterable, padvalue=None):
    """grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')
    原理是同一个迭代器的数组，采取的是轮询
    """
    return zip_longest(*[iter(iterable)] * n, fillvalue=padvalue)


def generate_walks(sc, G, num_paths, path_length,
                   num_workers=400, alpha=0, rand=random.Random(0)):
    def _generate_walks(args):
        nodes, num_paths, path_length, alpha, seed = args
        G = Graph().from_map(__current_graph.value)
        walks = []
        for walk in build_deepwalk_corpus_iter(G=G,
                                               nodes=nodes,
                                               num_paths=num_paths,
                                               path_length=path_length,
                                               alpha=alpha,
                                               rand=random.Random(seed)):
            walks.append(walk)
        return walks

    nodes = list(G.nodes())
    rand.shuffle(nodes)
    __current_graph = sc.broadcast(dict(G))  # since pickle cannot serialize defaultdict
    args_list = []
    nodes_per_worker = [list(filter(lambda z: z is not None, [y for y in x]))
                        for x in grouper(int(math.ceil(len(nodes) / num_workers)), nodes)]
    for npw in nodes_per_worker:
        args_list.append((npw, num_paths, path_length, alpha, rand.randint(0, 2 ** 31)))
    args_rdd = sc.parallelize(args_list, numSlices=num_workers)
    walks_rdd = args_rdd.flatMap(_generate_walks)
    return walks_rdd


def build_deepwalk_corpus_iter(G, nodes, num_paths, path_length, alpha=0, rand=random.Random(0)):
    for cnt in range(num_paths):
        for node in nodes:
            yield G.random_walk(path_length, rand=rand, alpha=alpha, start=node)
