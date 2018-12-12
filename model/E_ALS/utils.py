#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/12 16:38
# @Author  : Yajun Yin
# @Note    :
from __future__ import division

import heapq


class Rating(object):
    """
    mimic two constructors.
    * Rating(1,2,4) => @Rating(userId: 1, itemId: 2, score: 4.0000)
    * Rating("1::2::4", '::') => @Rating(userId: 1, itemId: 2, score: 4.0000)
    """

    def __init__(self, *args):
        if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], str):
            delimiter = args[1]
            line = args[0].split(delimiter)
            userId, itemId, score = int(line[0]), int(line[1]), float(line[2])
            timestamp = None if len(line) == 3 else float(line[3])
        elif len(args) == 3 or len(args) == 4:
            userId, itemId, score = {args[0], args[1], args[2]}
            timestamp = None if len(args) == 3 else args[3]
        else:
            raise ValueError("argument invalid!")
        self.userId = userId
        self.itemId = itemId
        self.score = score
        self.timestamp = timestamp

    def __str__(self):
        if self.timestamp:
            return "@Rating(userId: %d, itemId: %d, score: %.4f, timestamp: %.2f)" % (
                self.userId, self.itemId, self.score, self.timestamp)
        else:
            return "@Rating(userId: %d, itemId: %d, score: %.4f)" % (self.userId, self.itemId, self.score)

    def __repr__(self):
        return self.__str__()


def TopKeyByValue(hashmap, topK, ignoreKeys=None):
    """Get the topK keys (by its value) of a map. Does not consider the keys which are in ignoreKeys."""
    topQueue = []
    ignoreKeys = [] if not ignoreKeys else list(ignoreKeys)

    for item, score in hashmap.items():
        if not item in ignoreKeys:
            heapq.heappush(topQueue, (item, score))

    topItems = heapq.nlargest(topK, topQueue, key=lambda p: p[1])
    topKeys = list(map(lambda p: p[0], topItems))
    return topKeys
