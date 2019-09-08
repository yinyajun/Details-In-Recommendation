#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/11 12:13
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import abc
import numpy as np
import time
import utils as ut


class TopKRecommender(abc):
    def __init__(self, trainMatrix, testRatings, topK=10, ignoreTrain=False):
        # Rating matrix of training set. Users by Items.
        self.trainMatrix = trainMatrix
        # Test ratings. For showing progress only.
        self.testRatings = testRatings
        # Position to cutoff.
        self.topK = topK

        self.userCount, self.itemCount = trainMatrix.shape

        self.ignoreTrain = ignoreTrain
        # eval
        self.hits = None
        self.ndcgs = None
        self.precs = None

    @abc.abstractclassmethod
    def predict(self, u, i):
        """Get the prediction score of user u on item i. To be overridden."""
        return 0.0

    @abc.abstractclassmethod
    def buildModel(self):
        """Build the model."""
        pass

    @abc.abstractclassmethod
    def updateModel(self, u, i):
        """
        Update the model with a new observation."""
        pass

    @abc.abstractclassmethod
    def loss(self):
        return 0.0

    def showProgress(self, iter, start):
        """
        Show progress (evaluation) with current model parameters.
        :param iter: Current iteration
        :param start: Starting time of iteration
        """
        end_iter = time.time()
        if self.userCount == len(self.testRatings):  # leave-1-out eval
            self.evaluate()
        else:  # global split
            self.evaluationOnline(10)
        end_eval = time.time()
        print("Iter=%d[%.2f] <loss, hr, ndcg, prec>: "
              "%.4f\t %.4f\t %.4f\t %.4f\t [%.2f]" % (iter, end_iter - start, self.loss(), float(np.mean(self.hits)),
                                                      float(np.mean(self.ndcgs)), float(np.mean(self.precs)),
                                                      end_eval - end_iter))

    def evaluationOnline(self, interval):
        """
        Online evaluation (global split) by simulating the testing stream.
        :param interval: Print evaluation result per X iteration.
        :return:
        """
        testCount = len(self.testRatings)
        self.hits = np.zeros(testCount)
        self.ndcgs = np.zeros(testCount)
        self.precs = np.zeros(testCount)

        # break down the results by number of user ratings of the test pair
        # todo:

    def evaluate(self):
        """
        Offline evaluation (leave-1-out) for each user.
        :return:
        """
        assert self.userCount == len(self.testRatings)
        for u in xrange(self.userCount):
            assert u == self.testRatings[u].userId

        self.hits = []
        self.ndcgs = []
        self.precs = []

        # Run the evaluation
        for u in xrange(self.userCount):
            res = self.evaluate_for_user(u, self.testRatings[u].itemId)
            self.hits.append(res[0])
            self.ndcgs.append(res[1])
            self.precs.append(res[2])

    def evaluate_for_user(self, u, gtItem):
        """
        Evaluation for a specific user with given GT item.
        :return: hit_ratio, ndcg, precision
        """
        result = []
        map_item_score = dict()
        # Get the score of the test item first.
        maxScore = self.predict(u, gtItem)

        # Early stopping if there are topK items larger than maxScore.
        countLarger = 0
        for i in xrange(self.itemCount):
            score = self.predict(u, i)
            map_item_score[i] = score

            if score > maxScore:
                countLarger += 1
            if countLarger > self.topK:  # early stopping
                return result

        # Selecting topK items (does not exclude train items).
        if self.ignoreTrain:
            rankList = ut.TopKeyByValue(map_item_score, self.topK, self.trainMatrix.getrow(u).nonzero()[1])
        else:
            rankList = ut.TopKeyByValue(map_item_score, self.topK)

        result.append(self.getHitRatio(rankList, gtItem))
        result.append(self.getNDCG(rankList, gtItem))
        result.append(self.getPrecision(rankList, gtItem))
        return result

    @staticmethod
    def getHitRatio(rankList, gtItem):
        return 1 if gtItem in rankList else 0

    @staticmethod
    def getNDCG(rankList, gtItem):
        if gtItem in rankList:
            position = np.where(rankList == gtItem)[0][0] + 1
            return np.reciprocal(np.log2(position + 1))  # index start from 0
        return 0

    @staticmethod
    def getPrecision(rankList, gtItem):
        if gtItem in rankList:
            position = np.where(rankList == gtItem)[0][0] + 1
            return np.reciprocal(float(position))
        return 0
