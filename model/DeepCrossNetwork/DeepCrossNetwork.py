#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/28 11:14
# @Author  : Yajun Yin
# @Note    : modify metric; change some initializer; modify cross op; change loss reduction(mean); change BN;
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import six
from collections import namedtuple
from tensorflow.python.estimator.canned.head import _get_weights_and_check_match_logits
from tensorflow.python.ops.init_ops import glorot_normal_initializer
from tensorflow.contrib.training import HParams
from tensorflow.python.training import learning_rate_decay
from tensorflow.python.estimator.canned.head import _indicator_labels_mean
from tensorflow.python.estimator.canned.head import _accuracy_baseline
from tensorflow.python.estimator.canned.head import _auc
from tensorflow.python.summary import summary
from tensorflow.python.ops import nn

_PredictionKeys = namedtuple("predictionKeys", ['PROBABILITIES', 'LOGISTIC', 'CLASSES_IDS'])('probabilities',
                                                                                             'logistic', 'class_ids')

_DECAY_METHOD_NAME = {'exponential_decay': learning_rate_decay.exponential_decay,
                      'piecewise_constant': learning_rate_decay.piecewise_constant,
                      'polynomial_decay': learning_rate_decay.polynomial_decay,
                      'cosine_decay': learning_rate_decay.cosine_decay,
                      'cosine_decay_restarts': learning_rate_decay.cosine_decay_restarts,
                      'noisy_linear_cosine_decay': learning_rate_decay.noisy_linear_cosine_decay, }


class DeepCrossNetwork(tf.estimator.Estimator):
    def __init__(self,
                 model_dir=None,
                 columns=None,
                 cross_layer_num=2,
                 dnn_hidden_units=None,
                 dnn_dropout=None,
                 config=None,
                 dnn_activation_fn=tf.nn.relu,
                 weight_column=None,
                 optimizer=None,
                 optimizer_spec=None,
                 batch_norm=True,
                 l2_reg=None,
                 learning_rate_spec=None
                 ):
        """An estimator for Deep Cross Network models. More details, please see https://arxiv.org/pdf/1708.05123.pdf

            :param model_dir: Directory to save model parameters, graph and etc.
            This can also be used to load checkpoints from the directory into a estimator
            to continue training a previously saved model.
            :param columns: An iterable containing all the feature columns used
                by the model. All items in the set must be instances of
                classes derived from `_DenseColumn` such as `numeric_column`,
                `embedding_column`,`bucketized_column`, `indicator_column`.
                You can wrap them with an `embedding_column` or `indicator_column`.
            :param cross_layer_num: Number of cross layer.
            :param dnn_hidden_units: List of hidden units per layer in the deep part.
                All layers are fully connected.
            :param dnn_dropout: When not None, the probability we will drop out
                a given coordinate.
            :param config: RunConfig object to configure the runtime settings.
            :param dnn_activation_fn: Activation function applied to each layer.
            :param weight_column: A string or a `_NumericColumn` created by
                `tf.feature_column.numeric_column` defining feature column representing
                weights. It is used to down weight or boost examples during training. It
                will be multiplied by the loss of the example. If it is a string, it is
                used as a key to fetch weight tensor from the `features`. If it is a
                `_NumericColumn`, raw tensor is fetched by key `weight_column.key`,
                then weight_column.normalizer_fn is applied on it to get weight tensor.
            :param optimizer: An type of `tf.Optimizer` used to apply gradients to
                the model.
            :param optimizer_spec: An dict containing hyper params of the optimizer.
            :param batch_norm: An boolean whether to use batch normalization.
            :param l2_reg: An float or None. l2_regularizer in deep part of model.
            :param learning_rate_spec: An dict containing hyper params of learning rate decay.
            :return: An tf.estimator.Estimator instance.
            """
        # include params and hyper params
        hparams = HParams(feature_columns=columns,
                          cross_layer_num=cross_layer_num,
                          hidden_units=dnn_hidden_units,
                          dnn_activation_fn=dnn_activation_fn,
                          optimizer=optimizer,
                          optimizer_spec=optimizer_spec,
                          weight_column=weight_column,
                          dnn_dropout=dnn_dropout,
                          batch_norm=batch_norm,
                          l2_reg=l2_reg,
                          learning_rate_spec=learning_rate_spec)

        # generate model_fn
        def _model_fn(features, labels, mode, params):
            """ Model function used in the estimator.
            :param features: (Dict) A mapping from key to tensors, input features to the model.
            :param labels: (Tensor) Labels tensor for training and evaluation.
            :param mode: (ModeKeys) Specifies if training, evaluation or prediction.
            :param params: (HParams) hyperparameters.
            :return: (EstimatorSpec) Model to be run by Estimator depends on different mode.
            """
            # get info from params
            _print_params_info(params)

            # calc logits and generate EstimatorSpec
            with tf.variable_scope("dcn_model"):
                logit_fn = _dcn_logit_fn_builder(params)
                logits = logit_fn(features=features, mode=mode)
                _add_layer_summary(logits, "dcn_logits")
                return _create_estimator_spec(features, mode, logits, params, labels)

        super(DeepCrossNetwork, self).__init__(model_fn=_model_fn, model_dir=model_dir, config=config, params=hparams)


def _dcn_logit_fn_builder(params):
    """ Function builder for a dcn logit_fn.
    :param params: Hparams
    :return: A logit_fn (see below).
    """

    def dcn_logits_fn(features, mode):
        with tf.variable_scope('input_from_feature_columns'):
            input_layer = tf.feature_column.input_layer(features=features, feature_columns=params.feature_columns)
            column_num = input_layer.get_shape().as_list()[1]
            params.add_hparam("column_num", column_num)

            # build cross and deep part, respectively
            cross_output = _cross_architecture(input_layer, params)
            deep_output = _deep_architecture(input_layer, params, mode)

        with tf.variable_scope("logits"):
            # concat cross and deep outputs, generate logits
            m_layer = tf.concat([cross_output, deep_output], -1)
            logits = tf.layers.dense(inputs=m_layer, units=1, activation=None)

            return logits

    return dcn_logits_fn


def _create_estimator_spec(features, mode, logits, params, labels=None):
    """ Returns an `EstimatorSpec`.
    :param features: (Dict) A mapping from key to tensors, input features to the model.
    :param mode: (ModeKeys) Specifies if training, evaluation or prediction.
    :param logits: (Tensor) with shape `[batch_size, 1]`.
    :param params: (HParams) hyperparameters.
    :param labels: Labels integer or string `Tensor` with shape matching `logits`
    :return: (EstimatorSpec) Model to be run by Estimator depends on different mode
    """
    with tf.name_scope("calc_predictions"):
        with tf.name_scope("predictions"):
            pred_keys = _PredictionKeys
            logistic = tf.sigmoid(logits, name=pred_keys.LOGISTIC)
            two_class_logits = tf.concat((tf.zeros_like(logits), logits), axis=-1,
                                         name="two_class_logits")  # softmax([0,x])=sigmoid(x)
            probabilities = tf.nn.softmax(two_class_logits, name=pred_keys.PROBABILITIES)
            class_ids = tf.argmax(two_class_logits, axis=-1, name=pred_keys.CLASSES_IDS)
            class_ids = tf.expand_dims(class_ids, axis=-1)
            predictions = {pred_keys.PROBABILITIES: probabilities,
                           pred_keys.LOGISTIC: logistic,
                           pred_keys.CLASSES_IDS: class_ids
                           }
        # Predict
        if mode == tf.estimator.ModeKeys.PREDICT:
            # output = tf.estimator.export.PredictOutput(predictions)
            # use classificationOutput to consistent with serving.
            output = tf.estimator.export.ClassificationOutput(scores=probabilities)
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions,
                                              export_outputs={'serving_default': output})

        weighted_loss, unweighted_loss, weights, labels = _create_loss(features=features, params=params, logits=logits,
                                                                       labels=labels)

        if params.l2_reg:
            l2_loss = tf.losses.get_regularization_loss()
            weighted_loss += l2_loss

        # Eval
        metrics = _get_metric_op(labels, logistic, class_ids, weights, unweighted_loss)
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions,
                                              loss=weighted_loss,
                                              eval_metric_ops=metrics)

        # Train
        train_op = _get_train_op_fn(weighted_loss, params)
        lr = _learning_rate_decay(params)
        example_weight_sum = tf.reduce_sum(weights * tf.ones_like(unweighted_loss))
        mean_loss = weighted_loss / example_weight_sum
    with tf.name_scope(''):
        # if is losses.Reduction.SUM
        tf.summary.scalar('loss', weighted_loss)
        tf.summary.scalar('average_loss', mean_loss)
        tf.summary.scalar('learning_rate', lr)
    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=weighted_loss,
                                      train_op=train_op)


def _create_loss(features, params, logits, labels):
    # follows the logic of tf.losses.sigmoid_cross_entropy
    labels = tf.to_float(labels)

    # unweighted loss
    unweighted_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

    # calc weighted loss
    weight_column = params.weight_column
    weights = _get_weights_and_check_match_logits(
        features=features, weight_column=weight_column, logits=logits)  # get weight col tensor
    weighted_loss = tf.losses.compute_weighted_loss(unweighted_loss, weights=weights,
                                                    reduction=tf.losses.Reduction.MEAN)
    return weighted_loss, unweighted_loss, weights, labels


def _print_params_info(params):
    """
    Print model info.
    :param params: (HParams) hyperparameters.
    :return:
    """
    print('-' * 90)
    print("DeepCross Network Model Info.")
    for k, v in params.values().items():
        if k == "feature_columns":  # it is too long
            continue
        if k == 'dnn_activation_fn':
            v = v.__name__
        if k == 'optimizer':
            if isinstance(v, type):
                v = v.__name__
            else:
                raise ValueError('Optimizer must be an Optimizer class, Given:{}'.format(v))
        if k == 'learning_rate_spec':
            if not isinstance(v, dict):
                raise ValueError('learning_rate_spec must be a dict.')
            try:
                method_str = v['decay_method']
            except KeyError:
                method_str = 'None'
                v['method_str'] = method_str

        print(k + ":", v)
    print('-' * 90)


def _add_layer_summary(value, tag):
    summary.scalar('%s/fraction_of_zero_values' % tag, nn.zero_fraction(value))
    summary.histogram('%s/activation' % tag, value)


def _get_train_op_fn(loss, params):
    """
    Get the training Op.
    :param loss: (Tensor) Scalar Tensor that represents the loss function.
    :param params: (HParams) Hyperparameters (needs to have `optimizer`)
    :return: Training Op
    """
    # get optimizer params
    lr = _learning_rate_decay(params)
    opt_spec = params.optimizer_spec
    if not isinstance(opt_spec, dict):
        raise ValueError('optimizer_spec must be a dict.')
    try:
        opt_spec.pop('learning_rate')
    except KeyError:
        pass
    opt_spec['learning_rate'] = lr

    optimizer = params.optimizer(**opt_spec)
    grads, variables = zip(*optimizer.compute_gradients(loss))
    grads = [None if gradient is None else tf.clip_by_norm(gradient, 100.0) for gradient in grads]

    update_ops = tf.get_collection(
        tf.GraphKeys.UPDATE_OPS)  # be sure to add any batch_normalization ops before getting the update_ops collection.
    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(zip(grads, variables), global_step=tf.train.get_global_step())
    return train_op


def _get_metric_op(labels, logistic, class_ids, weights, unweighted_loss):
    """
    Return a dict of the metric Ops.
    :param labels: (Tensor) Labels tensor for training and evaluation.
    :param logistic: (Tensor) Predictions can be generated from it directly and auc metric also need it.
    :param class_ids:(Tensor)
    :param weights: (Tensor) Weight column tensor, used to calc weighted eval metric.
    :param unweighted_loss(Tensor)
    :return: Dict of metric results keyed by name.
    """
    with tf.name_scope(None, "metrics"):
        labels_mean = _indicator_labels_mean(labels=labels, weights=weights, name="label/mean")
        average_loss = tf.metrics.mean(unweighted_loss, weights, name="average_loss")
        accuracy = tf.metrics.accuracy(labels, class_ids, weights=weights, name="accuracy")
        precision = tf.metrics.precision(labels, class_ids, weights, name="precision")
        recall = tf.metrics.recall(labels, class_ids, weights, name="recall")
        accuracy_baseline = _accuracy_baseline(labels_mean)
        auc = _auc(labels, logistic, weights, name="auc")

        metric_op = {'average_loss': average_loss,
                     'accuracy': accuracy,
                     'precision': precision,
                     'recall': recall,
                     'accuracy_baseline': accuracy_baseline,
                     'auc': auc}

        return metric_op


def _cross_variable_creat(cross_layers_num, column_num):
    """
    Create variable for cross part
    :param cross_layers_num:(int)
    :param column_num: (Tensor) the dimension of input layer.
    :return: Created variable
    """
    w = tf.get_variable(name="cross_w", shape=[cross_layers_num, column_num],
                        initializer=tf.truncated_normal_initializer(0.0, 0.1), dtype=tf.float32)
    b = tf.get_variable(name="cross_b", shape=[cross_layers_num, column_num],
                        initializer=tf.truncated_normal_initializer(0.0, 0.1), dtype=tf.float32)
    return w, b


def _cross_op(x0, x, w, b):
    """
    Fits the residual of x_(k+1) - x(k).
    :param x0: Input of cross 0-th layer. (m*d)
    :param x: Output of cross k-th layer. (Also k+1_th layer input) (m*d)
    :param w: Weight parameters of the k_th layer. (d)
    :param b: Bias parameters of the k_th layer. (d)
    :return: The residual of output of cross k+1_th layer subtracts k_th layer.
    """
    x_w = tf.tensordot(x, w, axes=1)  # (m,)
    y = x0 * tf.expand_dims(x_w, -1) + b + x  # (m,d) broadcast
    return y


def _cross_architecture(input_layer, params):
    """
    Return the output operation following the cross part of network.
    :param input_layer: (Tensor) Input
    :param params: (HParams) Hyperparameters (needs to have "column_num")
    :return: Output Op for the cross part.
    """
    column_num = params.column_num
    cross_layer_num = params.cross_layer_num

    w, b = _cross_variable_creat(cross_layer_num, column_num)
    x0 = input_layer
    xl = x0
    for layer_id in range(cross_layer_num):
        with tf.variable_scope("cross_layer_%d" % layer_id):
            xl = _cross_op(x0, xl, w[layer_id], b[layer_id])
        _add_layer_summary(xl, "cross_layer_%d" % layer_id)
    return xl


def _deep_architecture(input_layer, params, mode):
    """
    Return the output operation following the deep part of network.
    :param input_layer: (Tensor) Input
    :param params: (HParams) Hyperparameters (needs to have "hidden_units", "dropout", "batch_norm", "l2_reg",
    "dnn_activation_fn")
    :param mode: (ModeKeys) Specifies if training, evaluation or prediction. Dropout and BN need it.
    :return: Output Op for the deep part.
    """
    hidden_units = params.hidden_units
    dropout = params.dnn_dropout
    is_bn = params.batch_norm
    l2_reg = params.l2_reg

    net = input_layer

    # regularizer(deep)
    if l2_reg:
        regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
    else:
        regularizer = None

    for layer_id, num_hidden in enumerate(hidden_units):
        with tf.variable_scope('hidden_layer_%d' % layer_id) as hidden_layer_scope:
            net = tf.layers.dense(inputs=net,
                                  units=num_hidden,
                                  activation=params.dnn_activation_fn,
                                  kernel_initializer=glorot_normal_initializer(),
                                  kernel_regularizer=regularizer,
                                  name=hidden_layer_scope)
            # use Batch Normalization(last layer use no BN, BN after relu)
            if is_bn and layer_id < len(hidden_units) - 1:
                is_training = mode == tf.estimator.ModeKeys.TRAIN
                net = _batch_normalization(input=net, is_training=is_training, scope='bn_%d' % layer_id)

            # dropout
            if dropout is not None and mode == tf.estimator.ModeKeys.TRAIN:
                with tf.name_scope('dropout'):
                    net = tf.layers.dropout(net, rate=dropout, training=True, name="dropout")
        _add_layer_summary(net, 'hidden_layer_%d' % layer_id)
    return net


def _batch_normalization(input, is_training=True, scope=None):
    # https://github.com/tensorflow/tensorflow/issues/1122#issuecomment-231917736
    # need to update Op since uodate_collections is not None.
    # tf.cond need
    return tf.cond(tf.cast(is_training, tf.bool),
                   lambda: tf.contrib.layers.batch_norm(input, is_training=True, scope=scope),
                   lambda: tf.contrib.layers.batch_norm(input, is_training=False, reuse=True, scope=scope))


def _learning_rate_decay(params):
    # easy use learning rate decay.
    global_step = tf.train.get_global_step()
    lr_spec = params.learning_rate_spec
    # checked lr_spec is a dict in print_info.
    lr_spec['global_step'] = global_step

    def _get_lr(**kwargs):
        try:
            method_str = kwargs['decay_method']
            if method_str not in _DECAY_METHOD_NAME:
                raise ValueError('Unsupported learning rate name: {}. Supported names are: {}'.format(method_str, tuple(
                    sorted(six.iterkeys(_DECAY_METHOD_NAME)))))
            decay_method = _DECAY_METHOD_NAME[method_str]
            kwargs.pop('decay_method')
        except KeyError:
            decay_method = _no_decay
        lr = decay_method(**kwargs)
        return lr

    try:
        ret = _get_lr(**lr_spec)
    except TypeError:
        raise TypeError('{} argument are not correct. Check again and note that global_step is omitted.'.format(
            params.learning_rate_spec['decay_method']))
    return ret


def _no_decay(**kwargs):
    # when learning rate need no decay.
    try:
        lr = kwargs['learning_rate']
    except KeyError:
        raise KeyError('learning rate must be provided in no_decay.')
    return lr
