#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/11 12:13

from abc import abstractclassmethod
import numpy as np


class TopKRecommender(object):
    def __init__(self, trainMatrix, testRatings, topK):
        # Rating matrix of training set. Users by Items.
        self.trainMatrix = trainMatrix
        # Test ratings. For showing progress only.
        self.testRatings = testRatings
        # Position to cutoff.
        self.topK = topK

        self.userCount, self.itemCount = trainMatrix.shape

    @abstractclassmethod
    def predict(self, u, i):
        """Get the prediction score of user u on item i. To be overridden."""
        pass

    @abstractclassmethod
    def buildModel(self):
        """Build the model."""
        pass

    @abstractclassmethod
    def updateModel(self, u, i):
        """
        Update the model with a new observation."""
        pass

    def showProgress(self, iter, start):
        """
        Show progress (evaluation) with current model parameters.
        :param iter: Current iteration
        :param start: Starting time of iteration
        """
        pass

    def evaluationOnline(self, interval):
        """
        Online evaluation (global split) by simulating the testing stream.
        :param interval: Print evaluation result per X iteration.
        :return:
        """
        pass

    def evaluate(self):
        """
        Offline evaluation (leave-1-out) for each user.
        :return:
        """
        pass

    def evaluate_for_user(self, u, gtItem):
        """
        Evaluation for a specific user with given GT item.
        :return: hit_ratio, ndcg, precision
        """
        pass

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
