#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/11 10:58
from __future__ import division

from TopKRecommender import TopKRecommender
from six.moves import xrange
from scipy import sparse as sp
import time
import numpy as np


class FastALS(TopKRecommender):
    def __init__(self,
                 trainMatrix,
                 testRatings,
                 topK,
                 factors=10,  # number of latent factors.
                 maxIter=500,  # maximum iterations.
                 maxIterOnline=1,  # maximum iterations for update model.
                 w0=1,  #
                 reg=0.01,  # regularization parameters
                 alpha=1,
                 showProgress=True,
                 showLoss=True,
                 ):
        # model priors to set.
        super().__init__(trainMatrix, testRatings, topK)
        self.factors = factors
        self.maxIter = maxIter
        self.maxIterOnline = maxIterOnline
        self.w0 = w0
        self.reg = reg
        self.alpha = alpha
        self.show_loss = showLoss
        self.show_progress = showProgress

        # model parameters to learn
        self.U = None  # latent vectors for users
        self.V = None  # latent vectors for items

        # caches
        self.SU = None
        self.SV = None
        self.prediction_users, self.prediction_items = None, None
        self.rating_users, self.rating_items = None, None
        self.w_users, self.w_items = None, None

        self.W = None  # weight for each positive instance in trainMatrix.
        self.Wi = None  # weight for negative instances on item i.
        self.w_new = 1  # weight for new instance in online learning

    def _negative_weight(self):
        """Set the Wi as a decay function w0 * pi ^ alpha"""
        p = np.zeros(self.itemCount)
        for i in xrange(self.itemCount):
            p[i] = self.trainMatrix.getcol(i).count_nonzero()

        # convert p[i] to probability
        p = p / sum(p)
        p = p ** self.alpha

        # assign weight
        self.Wi = self.w0 * p / sum(p)

        # By default, the weight for positive instance is uniformly 1.
        assert self.W.format == 'csc'
        row = self.W.indices
        col = self.W.tocsr().indices
        assert len(row) == len(col)
        data = np.ones(len(row))
        self.W = sp.csr_matrix((data, (row, col)))

    def _init_caches(self):
        self.prediction_users = 0
        # todo

    def _init_model_parameters(self):
        mean = 0
        std = 0.01
        self.U = np.random.normal(mean, std, (self.userCount, self.factors))
        self.V = np.random.normal(mean, std, (self.itemCount, self.factors))

        # init SU (S^p) as U^T * U
        self.SU = np.matmul(self.U.transpose(), self.U)
        # init SV (S^q) as V^T * diag(Wi) * V
        self.SV = np.matmul(np.matmul(self.V.transpose(), np.diag(self.Wi)), self.V)

    def buildModel(self):
        loss_pre = 10 ^ 5
        for iter in xrange(self.maxIter):
            start = time.time()

            # Update user latent vectors
            for u in xrange(self.userCount):
                self.update_user(u)

            # Update item latent vectors
            for i in xrange(self.itemCount):
                self.update_item(i)

            # Show progress
            if self.show_progress:
                self.showProgress(iter, start)

            # Show loss
            if self.show_loss:
                loss_pre = self.showLoss(iter, start, loss_pre)

    def runOneIteration(self):
        # Update user latent vectors
        for u in xrange(self.userCount):
            self.update_user(u)

        # Update item latent vectors
        for i in xrange(self.itemCount):
            self.update_item(i)

    def update_user(self, u):
        itemList = self.trainMatrix.getrow(u).nonzero()[1]
        if len(itemList) == 0:
            return
        # prediction cache for the user
        for i in itemList:
            self.prediction_items[i] = self.predict(u, i)
            self.rating_items[i] = self.trainMatrix[u, i]
            self.w_items[i] = self.W[u, i]

        oldVector = self.U.getrow(u).toarray()



    def update_item(self, i):
        pass

    def showLoss(self, iter, start, loss_pre):
        return 1

    def predict(self, u, i):
        predict = np.sum(self.U.getrow(u).multiply(self.V.getrow(i)))
        return predict

    def loss(self):
        """Fast way to calculate the loss function"""
        pass

    def updateModel(self, u, i):
        if self.trainMatrix[u, i] == 0:
            self._change_entry_in_sparse_matrix(self.trainMatrix, u, i, 1)
        if self.W[u, i] == 0:
            self._change_entry_in_sparse_matrix(self.W, u, i, self.w_new)

        # new item
        if self.Wi[i] == 0:
            self.Wi[i] = self.w0 / self.itemCount
            # Update the SV cache
            v_i = self.V.getrow(i).toarray()
            self.SV += self.Wi[i] * np.matmul(v_i.transpose(), v_i)

        # old item
        for iter in xrange(self.maxIterOnline):
            self.update_user(u)
            self.update_item(i)

    @staticmethod
    def _change_entry_in_sparse_matrix(sparse_matrix, i, j, value):
        # lil_matrix format is suitable for changing element
        tmp = sparse_matrix.tolil()
        tmp[i, j] = value
        return tmp.tocsr()
