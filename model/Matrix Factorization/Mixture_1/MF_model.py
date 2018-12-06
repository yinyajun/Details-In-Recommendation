#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/12 16:50
# @Author  : Yajun Yin
# @Note    :

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.estimator.canned.head import _auc
from tensorflow.python.estimator.canned.head import _indicator_labels_mean
from tensorflow.python.estimator.canned.head import _accuracy_baseline
from tensorflow.python.training import training


class BaseMFModel(tf.estimator.Estimator):
    """
    Factorize the matrix R (u, v) into P (u, k) and Q (k, v) low rank matrices:

       R[i,j] = P[i,:] * Q[:,j]

    Additional intercepts b0, bi, bj can be included, leading to the following model:

       R[i,j] = b0 + bi[i] + bj[j] + P[i,:] * Q[:,j]

    The model is commonly used for collaborative filtering in recommender systems, where
    the matrix R contains of ratings by u users of v products. When users rate products
    using some kind of rating system (e.g. "likes", 1 to 5 stars), we are talking about
    explicit ratings (Koren et al, 2009). When ratings are not available and instead we
    use indirect measures of preferences (e.g. clicks, purchases), we are talking about
    implicit ratings (Hu et al, 2008). For implicit ratings we use modified model, where
    we model the indicator variable:
    we model the indicator variable:

       D[i,j] = 1 if R[i,j] > 0 else 0

    and define additional weights:

       C[i,j] = 1 + alpha * R[i, j]

    or log weights:

       C[i,j] = 1 + alpha * log(1 + R[i, j])

    The model is defined in terms of minimizing the loss function (squared, logistic) between
    D[i,j] indicators and the values predicted using matrix factorization, where the loss is
    weighted using the C[i,j] weights (see Hu et al, 2008 for details). When using logistic
    loss, the predictions are passed through the sigmoid function to squeze them into the
    (0, 1) range.
    """

    def __init__(self,
                 model_dir=None,
                 embedding_cols=None,
                 bias_cols=None,
                 reg_pen=None,
                 learning_rate=None,
                 is_implicit=False,
                 loss='mse',
                 intercepts=True,
                 optimizer='Adam',
                 weights=None,
                 weights_coef=0,
                 config=None):

        if loss not in ['mse', 'cross entropy']:
            raise ValueError("use 'mse' or 'cross entropy' loss")

        if optimizer not in ['Adam', 'Ftrl', None]:
            raise ValueError("use 'Adam' or 'Ftrl' optimizer or None")

        if weights not in ['log_weight', 'normal_weight', None]:
            raise ValueError("use 'log_weight' or 'normal_weight' weights or None")

        self.user_emb_col = embedding_cols[0]
        self.user_emb = None
        self.item_emb_col = embedding_cols[1]
        self.item_emb = None
        self.user_bias_col = bias_cols[0]
        self.user_bias = None
        self.item_bias_col = bias_cols[1]
        self.item_bias = None
        self.reg_pen = reg_pen
        self.learning_rate = learning_rate
        self.intercepts = intercepts
        self.is_implicit = is_implicit
        self.loss = loss
        self.optimizer = optimizer
        self.weights = weights
        self.weights_coef = weights_coef

        self._print_info()

        super(BaseMFModel, self).__init__(model_fn=self._model_fn, model_dir=model_dir, config=config)

    def _model_fn(self, features, labels, mode):
        with tf.name_scope("model_fn"):

            logits, preds = self._logit_fn(features=features)

            # Prediction
            with tf.name_scope("predictions"):

                if self.loss == "cross entropy":

                    two_class_logits = tf.concat((tf.zeros_like(logits), logits), axis=-1, name="two_class_logits")
                    probabilities = tf.nn.softmax(two_class_logits, name="probabilities")
                    class_ids = tf.argmax(probabilities, axis=-1, name="class_ids")
                    class_ids = tf.expand_dims(class_ids, axis=-1)
                    predictions = {"probabilities": probabilities,
                                   "logistic": preds,
                                   "class_ids": class_ids}
                    output = tf.estimator.export.ClassificationOutput(scores=probabilities)
                else:
                    predictions = {"scores": preds}
                    output = tf.estimator.export.ClassificationOutput(scores=preds)

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                                                  export_outputs={'serving_default': output})

            with tf.name_scope("labels"):
                # labels
                labels = tf.to_float(labels)
                base = 4.0
                labels = tf.cast(labels >= base, tf.float32)

                if self.is_implicit:
                    # D[i,j] = 1 if R[i,j] > 0 else 0
                    labels = tf.clip_by_value(labels, 0, 1, name="labels")

                    if self.weights == "log_weight":
                        weights_term = tf.log1p(labels)
                    elif self.weights == "normal_weight":
                        weights_term = tf.identity(labels)
                    else:
                        weights_term = 0.0
                else:
                    labels = tf.identity(labels, name='labels')
                    weights_term = 0.0
                weights = tf.add(1.0, self.weights_coef * weights_term, name='weights')

            labels = tf.expand_dims(labels, axis=-1)  # compatible with logits dimension, [?,1]

            # Eval
            with tf.name_scope("eval"):

                cost = self._get_loss(labels, preds, weights)

                if self.loss == 'cross entropy':
                    metrics = self._log_loss_metrics(labels, preds, class_ids, cost, weights)
                else:
                    metrics = self._mse_metrics(labels, preds, cost)

                if mode == tf.estimator.ModeKeys.EVAL:
                    return tf.estimator.EstimatorSpec(mode=mode, loss=cost, eval_metric_ops=metrics)
            # Train
            with tf.name_scope("train"):
                train_op = self._get_train_op(self.learning_rate, cost, self.optimizer)
                return tf.estimator.EstimatorSpec(mode=mode, loss=cost, train_op=train_op)

    def _logit_fn(self, features):

        with tf.variable_scope("inputs"):

            self.user_emb = tf.feature_column.input_layer(features=features, feature_columns=self.user_emb_col)
            self.item_emb = tf.feature_column.input_layer(features=features, feature_columns=self.item_emb_col)

            if self.intercepts:
                self.user_bias = tf.feature_column.input_layer(features=features, feature_columns=self.user_bias_col)
                self.item_bias = tf.feature_column.input_layer(features=features, feature_columns=self.item_bias_col)
                self.global_bias = tf.get_variable("global_bias", shape=[], dtype=tf.float32,
                                                   initializer=tf.zeros_initializer())

        with tf.variable_scope("logits"):

            basic_logits = tf.reduce_sum(tf.multiply(self.user_emb, self.item_emb), axis=1, name="basic_logits")
            basic_logits = tf.expand_dims(basic_logits, axis=-1)  # (?,) -> (?,1)
            # data flow in estimator should begin with batch_num

            if self.intercepts:
                biases = tf.add(self.user_bias, self.item_bias) + self.global_bias
                logits = tf.add(basic_logits, biases, name="logits")
            else:
                logits = tf.identity(basic_logits, name="logits")

            if self.loss == "cross entropy":
                preds = tf.sigmoid(logits, name='prediction')
            else:
                preds = tf.identity(logits, name='prediction')

            return logits, preds

    def _get_loss(self, labels, preds, weights):
        with tf.name_scope("loss"):

            # regularizer term
            reg_emb = tf.nn.l2_loss(self.user_emb) + tf.nn.l2_loss(self.item_emb)
            if self.intercepts:
                reg_bias = tf.nn.l2_loss(self.user_bias) + tf.nn.l2_loss(self.item_bias)
                reg = reg_emb + reg_bias
            else:
                reg = reg_emb
            reg = tf.multiply(reg, self.reg_pen, name="regularization")

            # cost
            if self.loss == "cross entropy":
                loss = tf.losses.log_loss(predictions=preds, labels=labels, weights=weights)
            else:
                loss = tf.losses.mean_squared_error(predictions=preds, labels=labels, weights=weights)

            cost = tf.add(loss, reg, name="cost")
            return cost

    @staticmethod
    def _log_loss_metrics(labels, preds, class_ids, loss, weights):
        with tf.name_scope("metrics"):
            labels_mean = _indicator_labels_mean(labels=labels, weights=weights, name="label/mean")
            average_loss = tf.metrics.mean(loss, name='average_loss')
            accuracy = tf.metrics.accuracy(labels, class_ids, weights=weights, name="accuracy")
            precision = tf.metrics.precision(labels, class_ids, weights=weights, name="precision")
            recall = tf.metrics.recall(labels, class_ids, weights=weights, name="recall")
            accuracy_baseline = _accuracy_baseline(labels_mean)
            auc = _auc(labels, preds, weights=weights, name="auc")
            metric_op = {'average_loss': average_loss,
                         'accuracy': accuracy,
                         'precision': precision,
                         'recall': recall,
                         'accuracy_baseline': accuracy_baseline,
                         'auc': auc}
            return metric_op

    @staticmethod
    def _mse_metrics(labels, preds, loss):
        with tf.name_scope("metrics"):
            average_loss = tf.metrics.mean(loss, name='average_loss')
            rmse = tf.metrics.root_mean_squared_error(labels, preds, name='rmse')
            metric_op = {"average_loss": average_loss,
                         "rmse": rmse}
            return metric_op

    @staticmethod
    def _get_train_op(lr, loss, optimizer):

        if optimizer == "Ftrl":
            opt = tf.train.FtrlOptimizer(lr)
        elif optimizer == 'Adam':
            opt = tf.train.AdamOptimizer(lr)
        else:
            opt = tf.train.GradientDescentOptimizer(lr)

        train_op = opt.minimize(loss, global_step=tf.train.get_global_step())

        return train_op

    def _print_info(self):
        tf.logging.info('*' * 60)
        tf.logging.info('> reg_pen: %.2f' % self.reg_pen)
        tf.logging.info('> learning_rate: %.2f' % self.learning_rate)
        tf.logging.info('> intercepts: %s' % self.intercepts)
        tf.logging.info('> is_implicit: %s' % self.is_implicit)
        tf.logging.info('> loss: %s' % self.loss)
        tf.logging.info('> optimizer: %s' % self.optimizer)
        tf.logging.info('> weights: %s' % self.weights)
        tf.logging.info('> weight_coef: %.2f' % self.weights_coef)
        tf.logging.info('*' * 60)
