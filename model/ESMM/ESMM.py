#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops.init_ops import glorot_normal_initializer
from tensorflow.contrib.training import HParams
from tensorflow.python.estimator.canned.head import _indicator_labels_mean
from tensorflow.python.estimator.canned.head import _accuracy_baseline
from tensorflow.python.estimator.canned.head import _auc
from tensorflow.python.estimator.canned.head import _get_weights_and_check_match_logits
from tensorflow.python.ops import clip_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

_EPSILON = 1e-7


class ESMM(tf.estimator.Estimator):
    def __init__(self,
                 model_dir=None,
                 columns=None,
                 ctr_weight_column=None,
                 ctcvr_weight_column=None,
                 dnn_hidden_units=None,
                 dnn_dropout=None,
                 config=None,
                 dnn_activation_fn=tf.nn.relu,
                 optimizer=None):
        """
        Estimator for ESMM.
        :param model_dir: model dir.
        :param columns: deep columns since base model is dnn.
        :param ctr_weight_column: A string or a `_NumericColumn` created by
                `tf.feature_column.numeric_column` defining feature column representing
                weights. It is used to down weight or boost examples during training. It
                will be multiplied by the loss of the example. If it is a string, it is
                used as a key to fetch weight tensor from the `features`. If it is a
                `_NumericColumn`, raw tensor is fetched by key `weight_column.key`,
                then weight_column.normalizer_fn is applied on it to get weight tensor.
                Weight column corresponding to click label.
        :param ctcvr_weight_column: Weight column corresponding to convert label.
        :param dnn_hidden_units:
        :param dnn_dropout:
        :param config:
        :param dnn_activation_fn:
        :param optimizer:
        """
        hparams = HParams(feature_columns=columns,
                          ctr_weight_column=ctr_weight_column,
                          ctcvr_weight_column=ctcvr_weight_column,
                          hidden_units=dnn_hidden_units,
                          dnn_activation_fn=dnn_activation_fn,
                          dnn_dropout=dnn_dropout,
                          optimizer=optimizer)
        super(ESMM, self).__init__(model_fn=_model_fn, model_dir=model_dir, config=config, params=hparams)


def _model_fn(features, labels, mode, params):
    with tf.variable_scope('esmm'):
        with tf.variable_scope('ctr_model'):
            ctr_logits = _base_model(features, mode, params)
        with tf.variable_scope('cvr_model'):
            cvr_logits = _base_model(features, mode, params)

        # Prediction
        with tf.name_scope('predictions'):
            ctr_logistic = tf.sigmoid(ctr_logits, name='ctr_logistic')
            cvr_logistic = tf.sigmoid(cvr_logits, name='cvr_logistic')
            ctcvr_logistic = tf.multiply(ctr_logistic, cvr_logistic, name='ctcvr_logistic')
            # convert probabilities to logits considering numerical stability
            epsilon_ = ops.convert_to_tensor(_EPSILON, dtype=ctcvr_logistic.dtype.base_dtype)
            ctcvr_prob = clip_ops.clip_by_value(ctcvr_logistic, epsilon_, 1 - epsilon_)
            ctcvr_logits = math_ops.log(ctcvr_prob / (1 - ctcvr_prob))
            logits = {'ctr_logits': ctr_logits, 'ctcvr_logits': ctcvr_logits}
            logistic = {'ctr_logistic': ctr_logistic, 'ctcvr_logistic': ctcvr_logistic}
            #
            ctr_two_class_logits = tf.concat((tf.zeros_like(ctr_logits), ctr_logits), axis=-1,
                                             name="ctr_two_class_logits")
            ctr_class_ids = tf.argmax(ctr_two_class_logits, axis=-1, name='ctr_class_ids')
            ctr_class_ids = tf.expand_dims(ctr_class_ids, axis=-1)
            ctcvr_two_class_logits = tf.concat((tf.zeros_like(ctcvr_logits), ctcvr_logits), axis=-1,
                                               name="ctcvr_two_class_logits")
            ctcvr_class_ids = tf.argmax(ctcvr_two_class_logits, axis=-1, name='ctcvr_class_ids')
            ctcvr_class_ids = tf.expand_dims(ctcvr_class_ids, axis=-1)
            class_ids = {'ctr_class_ids': ctr_class_ids, 'ctcvr_class_ids': ctcvr_class_ids}
            probabilities = tf.nn.softmax(ctcvr_two_class_logits, name='probabilities')
            predictions = {"probabilities": probabilities,
                           "logistic": ctcvr_logistic,
                           "class_ids": ctcvr_class_ids}

        if mode == tf.estimator.ModeKeys.PREDICT:
            output = tf.estimator.export.ClassificationOutput(scores=probabilities)
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions,
                                              export_outputs={'serving_default': output})

        # labels dimension treatment
        labels["click_label"] = tf.expand_dims(labels["click_label"], axis=-1)
        labels["convert_label"] = tf.expand_dims(labels["convert_label"], axis=-1)
        weighted_loss, unweighted_loss, weights, labels = _get_loss(features, labels, logits, params)
        # Eval
        metrics = _get_metrics(labels, logistic, class_ids, unweighted_loss, weights)
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions,
                                              loss=weighted_loss,
                                              eval_metric_ops=metrics)
        # Train
        assert mode == tf.estimator.ModeKeys.TRAIN

        ctr_example_weight_sum = tf.reduce_sum(weights['click_weight'] * tf.ones_like(unweighted_loss))
        ctcvr_example_weight_sum = tf.reduce_sum(weights['convert_weight'] * tf.ones_like(unweighted_loss))
        ctr_mean_loss = weighted_loss / ctr_example_weight_sum
        ctcvr_mean_loss = weighted_loss / ctcvr_example_weight_sum

        tf.summary.scalar('train_ctr_mean_loss', ctr_mean_loss)
        tf.summary.scalar('train_ctcvr_mean_loss', ctcvr_mean_loss)
        tf.summary.scalar('train_loss', weighted_loss)
        tf.summary.scalar('train_ctr_acc', metrics['ctr_accuracy'][1])
        tf.summary.scalar('train_ctcvr_acc', metrics['ctcvr_accuracy'][1])
        optimizer = params.optimizer
        train_op = optimizer.minimize(weighted_loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=weighted_loss, train_op=train_op)


def _base_model(features, mode, params):
    """base model is DNN"""
    hidden_units = params.hidden_units
    dropout = params.dnn_dropout

    net = tf.feature_column.input_layer(features=features, feature_columns=params.feature_columns)
    for layer_id, num_hidden in enumerate(hidden_units):
        with tf.variable_scope('hiddenlayer_%d' % layer_id) as hidden_layer_scope:
            net = tf.layers.dense(inputs=net,
                                  units=num_hidden,
                                  activation=params.dnn_activation_fn,
                                  kernel_initializer=glorot_normal_initializer(),
                                  name=hidden_layer_scope)
            if dropout is not None and mode == tf.estimator.ModeKeys.TRAIN:
                net = tf.layers.dropout(net, rate=dropout, training=True, name='dropout')
    # logits
    logits = tf.layers.dense(net, 1, activation=None)
    return logits


def _get_loss(features, labels, logits, params):
    ctcvr_label = tf.to_float(labels['convert_label'])
    ctr_label = tf.to_float(labels['click_label'])
    ctr_logits, ctcvr_logits = logits['ctr_logits'], logits['ctcvr_logits']

    # unweighted loss
    unweighted_ctcvr_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=ctcvr_label, logits=ctcvr_logits,
                                                                    name='ctcvr_loss')
    unweighted_ctr_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=ctr_label, logits=ctr_logits, name='ctr_loss')
    unweighted_loss = tf.add(unweighted_ctr_loss, unweighted_ctcvr_loss, name='total_loss')

    # weighted loss
    ctr_weight_column = params.ctr_weight_column
    ctcvr_weight_column = params.ctcvr_weight_column
    ctr_weights = _get_weights_and_check_match_logits(features=features, weight_column=ctr_weight_column,
                                                      logits=ctr_logits)
    ctr_weighted_loss = tf.losses.compute_weighted_loss(unweighted_ctr_loss, weights=ctr_weights,
                                                        reduction=tf.losses.Reduction.MEAN)
    ctcvr_weights = _get_weights_and_check_match_logits(features=features, weight_column=ctcvr_weight_column,
                                                        logits=ctcvr_logits)
    ctcvr_weighted_loss = tf.losses.compute_weighted_loss(unweighted_ctcvr_loss, weights=ctcvr_weights,
                                                          reduction=tf.losses.Reduction.MEAN)
    weighted_loss = tf.add(ctr_weighted_loss, ctcvr_weighted_loss)
    labels = {'convert_label': ctcvr_label, 'click_label': ctr_label}
    weights = {'convert_weight': ctcvr_weights, 'click_weight': ctr_weights}
    return weighted_loss, unweighted_loss, weights, labels


def _get_metrics(labels, logistic, class_ids, unweighted_loss, weights):
    with tf.name_scope(None, 'metrics'):
        ctcvr_label, ctr_label = labels['convert_label'], labels['click_label']
        ctr_weight, ctcvr_weight = weights['click_weight'], weights['convert_weight']
        ctr_logistic, ctcvr_logistic = logistic['ctr_logistic'], logistic['ctcvr_logistic']
        ctr_class_ids, ctcvr_class_ids = class_ids['ctr_class_ids'], class_ids['ctcvr_class_ids']

        ctr_labels_mean = _indicator_labels_mean(labels=ctr_label, weights=ctr_weight, name="ctr_label/mean")
        ctcvr_labels_mean = _indicator_labels_mean(labels=ctcvr_label, weights=ctcvr_weight, name="ctcvr_label/mean")

        ctr_average_loss = tf.metrics.mean(unweighted_loss, ctr_weight, name='ctr_average_loss')
        ctcvr_average_loss = tf.metrics.mean(unweighted_loss, ctcvr_weight, name='ctcvr_average_loss')

        ctr_accuracy = tf.metrics.accuracy(ctr_label, ctr_class_ids, name='ctr_acc')
        ctcvr_accuracy = tf.metrics.accuracy(ctcvr_label, ctcvr_class_ids, name='ctcvr_acc')

        ctr_precision = tf.metrics.precision(ctr_label, ctr_class_ids, name="ctr_precision")
        ctcvr_precision = tf.metrics.precision(ctcvr_label, ctcvr_class_ids, name="cvr_precision")

        ctr_recall = tf.metrics.recall(ctr_label, ctr_class_ids, name="ctr_recall")
        ctcvr_recall = tf.metrics.recall(ctcvr_label, ctcvr_class_ids, name="cvr_recall")

        ctr_accuracy_baseline = _accuracy_baseline(ctr_labels_mean)
        ctcvr_accuracy_baseline = _accuracy_baseline(ctcvr_labels_mean)

        ctr_auc = _auc(ctr_label, ctr_logistic, name="ctr_auc")
        ctcvr_auc = _auc(ctcvr_label, ctcvr_logistic, name='ctcvr_auc')
        metric_op = {
            'ctr_average_loss': ctr_average_loss,
            'ctcvr_average_loss': ctcvr_average_loss,
            'ctr_accuracy': ctr_accuracy,
            'ctcvr_accuracy': ctcvr_accuracy,
            'ctr_precision': ctr_precision,
            'ctcvr_precision': ctcvr_precision,
            'ctr_recall': ctr_recall,
            'ctcvr_recall': ctcvr_recall,
            'ctr_accuracy_baseline': ctr_accuracy_baseline,
            'ctcvr_accuracy_baseline': ctcvr_accuracy_baseline,
            'ctr_auc': ctr_auc,
            'ctcvr_auc': ctcvr_auc}
        return metric_op
