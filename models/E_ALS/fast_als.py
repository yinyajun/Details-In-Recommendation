#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/11 10:58
from __future__ import division
from __future__ import print_function

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

        # weight
        self.W = sp.dok_matrix((self.userCount, self.itemCount))  # weight for each positive instance in trainMatrix.
        self.Wi = None  # weight for negative instances on item i.
        self.w_new = 1  # weight for new instance in online learning

        self._negative_weight()
        self._init_model_parameters()

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
        keys = self.trainMatrix.keys()
        for i, j in keys:
            self.W[i, j] = 1

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
        loss_pre = 10 ^ 8
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
        prediction_items = dict()
        for i in itemList:
            prediction_items[i] = self.predict(u, i)

        old_vector = self.U[u]
        for f in xrange(self.factors):
            numer = 0
            denom = 0
            # O(k) complexity for the negative part
            for k in xrange(self.factors):
                if k != f:
                    numer -= self.U[u, k] * self.SV[k, f]

            # O(Nu) complexity for the positive part
            for i in itemList:
                w_ui = self.W[u, i]
                r_ui = self.trainMatrix[u, i]
                c_i = self.Wi[i]
                r_ui_f = prediction_items[i] - self.U[u, f] * self.V[i, f]

                numer += (w_ui * r_ui - (w_ui - c_i) * r_ui_f) * self.V[i, f]
                denom += (w_ui - c_i) * self.V[i, f] * self.V[i, f]
            denom += self.SV[f, f] + self.reg

            # Parameter Update
            self.U[u, f] = numer / denom

        # Update the SU cache
        self.SU = self.SU - np.matmul(old_vector.transpose(), old_vector) + np.matmul(self.U[u].transpose(), self.U[u])

    def update_item(self, i):
        userList = self.trainMatrix.getcol(i).nonzero()[0]
        if len(userList) == 0:
            return

        # prediction cache for the item
        prediction_users = dict()
        for u in userList:
            prediction_users[u] = self.predict(u, i)  # use hash_map instead of array

        old_vector = self.V[i]
        for f in xrange(self.factors):
            numer = 0
            denom = 0
            # O(K) complexity for the negative part
            for k in xrange(self.factors):
                if k != f:
                    numer -= self.V[i, k] * self.SU[k, f]
            numer *= self.Wi[i]

            # O(Ni) complexity for the positive ratings part
            for u in userList:
                w_ui = self.W[u, i]
                r_ui = self.trainMatrix[u, i]
                c_i = self.Wi[i]
                r_ui_f = prediction_users[u] - self.U[u, f] * self.V[i, f]

                numer += (w_ui * r_ui - (w_ui - c_i) * r_ui_f) * self.U[u, f]
                denom += (w_ui - c_i) * self.U[u, f] * self.U[u, f]
            denom += self.SU[f, f] + self.reg

            # Parameter Update
            self.V[i, f] = numer / denom

        # Update the SV cache
        self.SV = self.SV \
                  - self.Wi[i] * np.matmul(old_vector.transpose(), old_vector) \
                  + self.Wi[i] * np.matmul(self.V[i].transpose(), self.V[i])

    def showLoss(self, iter, start, loss_pre):
        start1 = time.time()
        loss_cur = self.loss()
        symbol = '-' if loss_pre >= loss_cur else '+'
        print("Iter=%d [%.2f]\t [%s]loss: %.4f [%.2f]" % (iter, start1 - start, symbol, loss_cur, time.time() - start1))
        return loss_cur

    def predict(self, u, i):
        predict = float(np.inner(self.U[u], self.V[i]))
        return predict

    def loss(self):
        """Fast way to calculate the loss function"""
        # regularizer
        L = np.sum(self.U ** 2) + np.sum(self.V ** 2)
        L *= self.reg

        # loss = data loss + missing data loss(exploit cache)
        for u in xrange(self.userCount):
            loss = 0
            itemList = self.trainMatrix.getrow(u).nonzero()[1]
            for i in itemList:
                pred = self.predict(u, i)
                loss += self.W[u, i] * (pred - self.trainMatrix[u, i]) ** 2
                loss -= self.Wi[i] * (pred ** 2)
            loss += np.inner(np.matmul(self.U[u], self.SV), self.U[u])
            L += loss
        return float(L)

    def updateModel(self, u, i):
        if self.trainMatrix[u, i] == 0:
            self.trainMatrix[u, i] = 1
            self.W[u, i] = self.w_new

        # new item
        if self.Wi[i] == 0:
            self.Wi[i] = self.w0 / self.itemCount
            # Update the SV cache
            v_i = self.V[i]
            self.SV += self.Wi[i] * np.matmul(v_i.transpose(), v_i)

        # old item
        for iter in xrange(self.maxIterOnline):
            self.update_user(u)
            self.update_item(i)
