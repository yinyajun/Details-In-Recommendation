#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/7 15:43
# @Author  : Yajun Yin
# @Note    :

from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.estimator.canned.head import _auc
from tensorflow.contrib.training import HParams


class Svd(tf.estimator.Estimator):
    """svd for implicit feedback, i.e., add sigmoid function on the predictions.
    So, mse is replaced by cross_entropy error instead."""
    def __init__(self,
                 model_dir=None,
                 model_name=None,
                 user_emb=None,
                 item_emb=None,
                 l2_pen=None,
                 learning_rate=None,
                 config=None):

        self.user_emb = user_emb,
        self.item_emb = item_emb,
        self.l2_pen = l2_pen,
        self.learning_rate = learning_rate

        hparams = HParams(model_name=model_name)

        super(Svd, self).__init__(model_fn=self._model_fn,
                                  model_dir=model_dir,
                                  config=config,
                                  params=hparams)

    def _model_fn(self, features, labels, mode, params):
        with tf.variable_scope(params.model_name):
            logits = self.logit_fn(features=features)

            # Prediction
            with tf.name_scope("predictions"):
                logistic = tf.sigmoid(logits)
                two_class_logits = tf.concat((tf.zeros_like(logits), logits), axis=-1, name="two_class_logits")
                probabilities = tf.nn.softmax(two_class_logits, name="probabilities")
                class_ids = tf.argmax(probabilities, axis=-1, name="class_ids")
                class_ids = tf.expand_dims(class_ids, axis=-1)

                predictions = {"probabilities": probabilities,
                               "logistic": logistic,
                               "class_ids": class_ids}
            if mode == tf.estimator.ModeKeys.PREDICT:
                output = tf.estimator.export.ClassificationOutput(scores=probabilities)
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  predictions=predictions,
                                                  export_outputs={'serving_default': output})
            labels = tf.to_float(labels)
            labels = tf.expand_dims(labels, axis=-1)  # compatible with logits dimension, [?,1]
            loss = self._get_loss(labels, logits)
            # Eval
            metrics = self._get_metrics(labels, logistic, class_ids, loss)
            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  predictions=predictions,
                                                  loss=loss,
                                                  eval_metric_ops=metrics)
            # Train
            train_op = self._get_train_op(self.learning_rate, loss)
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op)

    def logit_fn(self, features):
        with tf.variable_scope("input_from_feature_columns"):
            self.user = tf.feature_column.input_layer(features=features, feature_columns=self.user_emb)
            self.item = tf.feature_column.input_layer(features=features, feature_columns=self.item_emb)
            logits = tf.reduce_sum(tf.multiply(self.user, self.item), axis=-1)
            logits = tf.expand_dims(logits, axis=-1)
            tf.assert_rank(logits, 2)
            return logits

    def _get_loss(self, labels, logits):
        logloss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        logloss = tf.reduce_mean(logloss)

        # regularizer term on user vct and prod vct
        reg = self.l2_pen * (tf.nn.l2_loss(self.user) + tf.nn.l2_loss(self.item))
        loss = logloss + reg
        return loss

    @staticmethod
    def _get_metrics(labels, logistic, class_ids, loss):
        with tf.name_scope(None, "metrics"):
            average_loss = tf.metrics.mean(loss, name='average_loss')
            accuracy = tf.metrics.accuracy(labels, class_ids, name="accuracy")
            precision = tf.metrics.precision(labels, class_ids, name="precision")
            recall = tf.metrics.recall(labels, class_ids, name="recall")
            auc = _auc(labels, logistic, name="auc")
            metric_op = {"average_loss": average_loss,
                         "accuracy": accuracy,
                         "precision": precision,
                         "recall": recall,
                         "auc": auc}
            return metric_op

    @staticmethod
    def _get_train_op(lr, loss):
        train_op = tf.train.GradientDescentOptimizer(lr). \
            minimize(loss, global_step=tf.train.get_global_step())
        return train_op


class BiasedSvd(Svd, tf.estimator.Estimator):
    def __init__(self,
                 model_dir=None,
                 model_name=None,
                 user_emb=None,
                 item_emb=None,
                 user_bias=None,
                 item_bias=None,
                 l2_pen=None,
                 learning_rate=None,
                 config=None):
        self.user_emb = user_emb,
        self.item_emb = item_emb,
        self.l2_pen = l2_pen,
        self.learning_rate = learning_rate
        self.user_bias = user_bias
        self.item_bias = item_bias

        super(BiasedSvd, self).__init__(model_dir=model_dir,
                                        model_name=model_name,
                                        user_emb=user_emb,
                                        item_emb=item_emb,
                                        l2_pen=l2_pen,
                                        learning_rate=learning_rate,
                                        config=config)

    def logit_fn(self, features):
        with tf.variable_scope("logits"):
            self.user = tf.feature_column.input_layer(features=features, feature_columns=self.user_emb)
            self.item = tf.feature_column.input_layer(features=features, feature_columns=self.item_emb)
            self.user_b = tf.feature_column.input_layer(features=features, feature_columns=self.user_bias)
            self.item_b = tf.feature_column.input_layer(features=features, feature_columns=self.item_bias)

            self.alpha = tf.get_variable('alpha', [], initializer=tf.constant_initializer(0.00000001, dtype=tf.float32),
                                         trainable=True)
            self.global_bias = tf.get_variable('global_bias', [1],
                                               initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                                               trainable=True)
            emb_logits = self.alpha * tf.reduce_sum(tf.multiply(self.user, self.item), axis=1)
            emb_logits = tf.expand_dims(emb_logits, axis=-1)
            bias = tf.add(self.user_b, self.item_b) + self.global_bias
            logits = emb_logits + bias

            tf.assert_equal(logits.shape[1], 1)  # logits.shape=[m,1]
            return logits

    def _get_loss(self, labels, logits):
        labels = tf.to_float(labels)
        logloss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        logloss = tf.reduce_mean(logloss)

        # regularizer term on user vct and prod vct
        reg = self.l2_pen * (tf.nn.l2_loss(self.user) + tf.nn.l2_loss(self.item))
        reg_bias = self.l2_pen * (tf.nn.l2_loss(self.user_b) + tf.nn.l2_loss(self.item_b))
        loss = logloss + reg + reg_bias
        return loss


