#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/10 16:14
# @Author  : Yajun Yin
# @Note    :
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import six
import math
from tensorflow.python.ops.init_ops import glorot_normal_initializer
from tensorflow.contrib.training import HParams
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.estimator import estimator
from tensorflow.python.estimator.canned.head import _get_weights_and_check_match_logits
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.estimator.canned import dnn
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.estimator.canned import linear
from tensorflow.python.estimator.canned import optimizers
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import losses
from tensorflow.python.summary import summary
from tensorflow.python.training import distribute as distribute_lib
from tensorflow.python.training import sync_replicas_optimizer
from tensorflow.python.training import training_util
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.estimator.canned.head import _indicator_labels_mean
from tensorflow.python.estimator.canned.head import _accuracy_baseline
from tensorflow.python.estimator.canned.head import _auc

_DNN_LEARNING_RATE = 0.001
_LINEAR_LEARNING_RATE = 0.005
_EPSILON = 1e-7


def _add_layer_summary(value, tag):
    summary.scalar('%s/fraction_of_zero_values' % tag, nn.zero_fraction(value))
    summary.histogram('%s/activation' % tag, value)


def _check_no_sync_replicas_optimizer(optimizer):
    if isinstance(optimizer, sync_replicas_optimizer.SyncReplicasOptimizer):
        raise ValueError(
            'SyncReplicasOptimizer does not support multi optimizers case. '
            'Therefore, it is not supported in DNNLinearCombined model. '
            'If you want to use this optimizer, please use either DNN or Linear '
            'model.')


def _linear_learning_rate(num_linear_feature_columns):
    """Returns the default learning rate of the linear model.

    The calculation is a historical artifact of this initial implementation, but
    has proven a reasonable choice.

    Args:
      num_linear_feature_columns: The number of feature columns of the linear
        model.

    Returns:
      A float.
    """
    default_learning_rate = 1. / math.sqrt(num_linear_feature_columns)
    return min(_LINEAR_LEARNING_RATE, default_learning_rate)


class ESMM_W_D(tf.estimator.Estimator):
    def __init__(self,
                 model_dir=None,
                 linear_feature_columns=None,
                 dnn_feature_columns=None,
                 linear_optimizer=None,
                 ctr_weight_column=None,
                 ctcvr_weight_column=None,
                 dnn_hidden_units=None,
                 dnn_dropout=None,
                 config=None,
                 dnn_optimizer=None,
                 input_layer_partitioner=None):
        hparams = HParams(linear_feature_columns=linear_feature_columns,
                          dnn_feature_columns=dnn_feature_columns,
                          linear_optimizer=linear_optimizer,
                          ctr_weight_column=ctr_weight_column,
                          ctcvr_weight_column=ctcvr_weight_column,
                          dnn_hidden_units=dnn_hidden_units,
                          dnn_dropout=dnn_dropout,
                          dnn_optimizer=dnn_optimizer,
                          input_layer_partitioner=input_layer_partitioner,
                          config=config)
        super(ESMM_W_D, self).__init__(model_fn=_model_fn, model_dir=model_dir, config=config, params=hparams)


def _model_fn(features, labels, mode, params):
    with tf.variable_scope('esmm'):
        with tf.variable_scope('ctr_model'):
            parent_scope_name = 'esmm/ctr_model'
            ctr_logits, ctr_train_op_fn = _base_model(features=features,
                                                      parent_scope_name=parent_scope_name,
                                                      mode=mode,
                                                      linear_feature_columns=params.linear_feature_columns,
                                                      linear_optimizer=params.linear_optimizer,
                                                      dnn_feature_columns=params.dnn_feature_columns,
                                                      dnn_optimizer=params.dnn_optimizer,
                                                      dnn_hidden_units=params.dnn_hidden_units,
                                                      dnn_dropout=params.dnn_dropout,
                                                      config=params.config)
        with tf.variable_scope('cvr_model'):
            parent_scope_name = 'esmm/cvr_model'
            cvr_logits, cvr_train_op_fn = _base_model(features=features,
                                                      parent_scope_name=parent_scope_name,
                                                      mode=mode,
                                                      linear_feature_columns=params.linear_feature_columns,
                                                      linear_optimizer=params.linear_optimizer,
                                                      dnn_feature_columns=params.dnn_feature_columns,
                                                      dnn_optimizer=params.dnn_optimizer,
                                                      dnn_hidden_units=params.dnn_hidden_units,
                                                      dnn_dropout=params.dnn_dropout,
                                                      config=params.config)
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
        def _train_op_fn(weighted_loss):
            ctr_train_op = ctr_train_op_fn(weighted_loss)
            cvr_train_op = cvr_train_op_fn(weighted_loss)
            global_step = training_util.get_global_step()
            with ops.control_dependencies([ctr_train_op, cvr_train_op]):
                return distribute_lib.increment_var(global_step)

        train_op = _train_op_fn(weighted_loss)

        ctr_example_weight_sum = tf.reduce_sum(weights['click_weight'] * tf.ones_like(unweighted_loss))
        ctcvr_example_weight_sum = tf.reduce_sum(weights['convert_weight'] * tf.ones_like(unweighted_loss))
        ctr_mean_loss = weighted_loss / ctr_example_weight_sum
        ctcvr_mean_loss = weighted_loss / ctcvr_example_weight_sum

        tf.summary.scalar('train_ctr_mean_loss', ctr_mean_loss)
        tf.summary.scalar('train_ctcvr_mean_loss', ctcvr_mean_loss)
        tf.summary.scalar('train_loss', weighted_loss)
        tf.summary.scalar('train_ctr_acc', metrics['ctr_accuracy'][1])
        tf.summary.scalar('train_ctcvr_acc', metrics['ctcvr_accuracy'][1])
        return tf.estimator.EstimatorSpec(mode=mode, loss=weighted_loss, train_op=train_op)


def _base_model(features=None,
                parent_scope_name=None,
                mode=None,
                linear_feature_columns=None,
                linear_optimizer='Ftrl',
                dnn_feature_columns=None,
                dnn_optimizer='Adagrad',
                dnn_hidden_units=None,
                dnn_activation_fn=nn.relu,
                dnn_dropout=None,
                input_layer_partitioner=None,
                config=None):
    if not isinstance(features, dict):
        raise ValueError('features should be a dictionary of `Tensor`s. '
                         'Given type: {}'.format(type(features)))
    if not linear_feature_columns and not dnn_feature_columns:
        raise ValueError(
            'Either linear_feature_columns or dnn_feature_columns must be defined.')

    num_ps_replicas = config.num_ps_replicas if config else 0
    input_layer_partitioner = input_layer_partitioner or (
        partitioned_variables.min_max_variable_partitioner(
            max_partitions=num_ps_replicas,
            min_slice_size=64 << 20))

    # Build DNN Logits.
    dnn_parent_scope = 'dnn'

    if not dnn_feature_columns:
        dnn_logits = None
    else:
        dnn_optimizer = optimizers.get_optimizer_instance(
            dnn_optimizer, learning_rate=_DNN_LEARNING_RATE)
        _check_no_sync_replicas_optimizer(dnn_optimizer)
        if not dnn_hidden_units:
            raise ValueError(
                'dnn_hidden_units must be defined when dnn_feature_columns is '
                'specified.')
        dnn_partitioner = (
            partitioned_variables.min_max_variable_partitioner(
                max_partitions=num_ps_replicas))
        with variable_scope.variable_scope(
                dnn_parent_scope,
                values=tuple(six.itervalues(features)),
                partitioner=dnn_partitioner):

            dnn_logit_fn = dnn._dnn_logit_fn_builder(  # pylint: disable=protected-access
                units=1,
                hidden_units=dnn_hidden_units,
                feature_columns=dnn_feature_columns,
                activation_fn=dnn_activation_fn,
                dropout=dnn_dropout,
                input_layer_partitioner=None)
            dnn_logits = dnn_logit_fn(features=features, mode=mode)

    linear_parent_scope = 'linear'

    if not linear_feature_columns:
        linear_logits = None
    else:
        linear_optimizer = optimizers.get_optimizer_instance(
            linear_optimizer,
            learning_rate=_linear_learning_rate(len(linear_feature_columns)))
        _check_no_sync_replicas_optimizer(linear_optimizer)
        with variable_scope.variable_scope(
                linear_parent_scope,
                values=tuple(six.itervalues(features)),
                partitioner=input_layer_partitioner) as scope:
            logit_fn = linear._linear_logit_fn_builder(  # pylint: disable=protected-access
                units=1,
                feature_columns=linear_feature_columns)
            linear_logits = logit_fn(features=features)
            _add_layer_summary(linear_logits, scope.name)

    # Combine logits
    if dnn_logits is not None and linear_logits is not None:
        logits = dnn_logits + linear_logits
    elif dnn_logits is not None:
        logits = dnn_logits
    else:
        logits = linear_logits

    def _train_op_fn(loss):
        """Returns the op to optimize the loss."""
        train_ops = []
        if dnn_logits is not None:
            train_ops.append(
                dnn_optimizer.minimize(
                    loss,
                    var_list=ops.get_collection(
                        ops.GraphKeys.TRAINABLE_VARIABLES,
                        scope=parent_scope_name + '/' + dnn_parent_scope)))
        if linear_logits is not None:
            train_ops.append(
                linear_optimizer.minimize(
                    loss,
                    var_list=ops.get_collection(
                        ops.GraphKeys.TRAINABLE_VARIABLES,
                        scope=parent_scope_name + '/' + linear_parent_scope)))

        train_op = control_flow_ops.group(*train_ops)
        with ops.control_dependencies([train_op]):
            return tf.no_op()

    return logits, _train_op_fn


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
