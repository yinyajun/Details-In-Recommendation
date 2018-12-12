#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/12 17:48
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import abc
import numpy as np
import time
import utils as ut



def ReadRatings_GlobalSplit(ratingFile, testRatio):
    userCount = 0
    itemCount = 0
    print("Global splitting with testRatio " + str(testRatio))

    # Step 1. Construct data structure for sorting.
    print("Read ratings and sort.")
    startTime = time.time()
    ratings = []
    with open(ratingFile, 'r') as f:
        for line in f:
            line = line.strip('\n')
            rating = ut.Rating(line, ',')
            ratings.append(rating)
            userCount = max(userCount, rating.userId)
            itemCount = max(itemCount, rating.itemId)

    # Step 2. Sort the ratings by time (small->large).
    ratings.sort()  # Rating class need __lt__ to sort asc
    print("[%.2f]" % time.time() - startTime)

    # Step 3. Generate trainMatrix and testStream
    startTime = time.time()
    trainMatrix = sp.dok_matrix(userCount, itemCount)
    testRatings = []

    testCount = int(len(ratings) * testRatio)
    trainCount = len(ratings) - testCount
    count = 0

    for rating in ratings:
        if count < trainCount:
            trainMatrix[rating.userId, rating.itemId] = 1
        else:
            testRatings.append(rating)
        count += 1

    # Count number of new users/items/ratings in the test data
    newUsers = []
    newRatings = 0
    for u in xrange(userCount):



def ReadRatings_HoldOneOut(ratingFile):
    pass


def deduplicate(ratingFile):
    pass


def evaluate_model(model, name):
    pass


def evaluate_model_online(model, name, interval):
    pass


def main():
    pass


if __name__ == '__main__':
