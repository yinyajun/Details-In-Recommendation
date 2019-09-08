#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/1 22:53
# @Author  : Yajun Yin
# @Note    :

import six
import tensorflow as tf
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.estimator.canned.optimizers import get_optimizer_instance
from tensorflow.python.estimator.model_fn import ModeKeys
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.framework import ops
from tensorflow.python.layers import normalization, core as core_layers
from tensorflow.python.ops import array_ops, gen_array_ops, init_ops, nn, \
    variable_scope, math_ops, control_flow_ops, partitioned_variables
from tensorflow.python.summary import summary
from tensorflow.python.training import training_util, sync_replicas_optimizer, distribute as distribute_lib
from tensorflow.python.feature_column.feature_column import _normalize_feature_columns, _LazyBuilder, _DenseColumn


def _add_hidden_layer_summary(value, tag):
    summary.scalar('%s/fraction_of_zero_values' % tag, nn.zero_fraction(value))
    summary.histogram('%s/activation' % tag, value)


def _add_layer_summary(value, tag):
    summary.scalar('%s/fraction_of_zero_values' % tag, nn.zero_fraction(value))
    summary.histogram('%s/activation' % tag, value)


def _compute_fraction_of_zero(cols_to_vars):
    all_weight_vars = []
    for var_or_var_list in cols_to_vars.values():
        # Skip empty-lists associated with columns that created no Variables.
        if var_or_var_list:
            all_weight_vars += [
                array_ops.reshape(var, [-1]) for var in var_or_var_list
            ]
    return nn.zero_fraction(array_ops.concat(all_weight_vars, axis=0))


def _check_no_sync_replicas_optimizer(optimizer):
    if isinstance(optimizer, sync_replicas_optimizer.SyncReplicasOptimizer):
        raise ValueError(
            'SyncReplicasOptimizer does not support multi optimizers case. '
            'Therefore, it is not supported in DNNLinearCombined model. '
            'If you want to use this optimizer, please use either DNN or Linear '
            'model.')


class DeepFM(tf.estimator.Estimator):
    # use categorical feature column to support multi-hot features
    # todo: support numerical feature column for fm_feature_columns
    def __init__(self,
                 model_dir=None,
                 linear_feature_columns=None,
                 linear_optimizer='Ftrl',
                 linear_sparse_combiner='sum',
                 dnn_feature_columns=None,
                 dnn_optimizer='Adagrad',
                 dnn_hidden_units=None,
                 dnn_activation_fn=nn.relu,
                 dnn_dropout=None,
                 fm_embedding_size=None,
                 n_classes=2,
                 weight_column=None,
                 label_vocabulary=None,
                 input_layer_partitioner=None,
                 config=None,
                 warm_start_from=None,
                 loss_reduction=tf.losses.Reduction.SUM,
                 batch_norm=False):
        """
        仿照wide_deep官方模型写法的deepFM模型，使用feature_column作特征处理。
        * 不用将数据集转换为libsvm格式.
        * 支持multi-hot类型特征，例如用户的历史消费物品序列.
        * 暂不支持数值型特征的交叉.
        【特征说明】
        参考train.py.
        1. 原始特征分为两种：
        * 数值特征(dense)：没啥好说的
        * 离散特征(sparse)：id类特征（vocabulary size特别大，使用特征hash），非id类特征（vocabulary size在几个到几百个之间，直接one-hot编码）
        特别的，当多个不定长离散特征组成一个序列，直接编码就是multi-hot特征，非常常见。
        2. 简要叙述官方wide_deep
        原生的wide部分由于其实现方式，直接支持sparse tensor的输入；
        而dnn部分只支持dense tensor输入。
        这里直接copy官方写法，也分为两部分输入。
        【DeepFM简要说明】
        * 按理说，deepFM的一阶项和二阶项应该都是相同的输入，分为两部分是仿照wide_deep写法，能够直接复用linear model代码。
        * 数值特征目前还不支持。
        * 将离散特征直接传给linear_feature_columns，作为一阶项的输入。
        * 将离散特征通过embedding变为dense tensor后传给dnn_feature_columns，作为二阶项和dnn的输入。
        * 实质上，一阶项和二阶项的输入是相同的。
        * 为了稍微利用数值特征，将数值特征分桶后用在了linear部分，而二阶项和dnn部分目前没有使用数值特征（todo）
        """

        linear_feature_columns = linear_feature_columns or []
        dnn_feature_columns = dnn_feature_columns or []
        self._feature_columns = (
            list(linear_feature_columns) +
            list(dnn_feature_columns)
        )
        if not self._feature_columns:
            raise ValueError('empty columns.')

        if n_classes == 2:
            head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(  # pylint: disable=protected-access
                weight_column=weight_column,
                label_vocabulary=label_vocabulary,
                loss_reduction=loss_reduction)
        else:
            head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(  # pylint: disable=protected-access
                n_classes,
                weight_column=weight_column,
                label_vocabulary=label_vocabulary,
                loss_reduction=loss_reduction)

        def _model_fn(features, labels, mode, config):
            return _DeepFM_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                head=head,
                linear_feature_columns=linear_feature_columns,
                linear_optimizer=linear_optimizer,
                linear_sparse_combiner=linear_sparse_combiner,
                dnn_feature_columns=dnn_feature_columns,
                dnn_optimizer=dnn_optimizer,
                dnn_hidden_units=dnn_hidden_units,
                dnn_activation_fn=dnn_activation_fn,
                dnn_dropout=dnn_dropout,
                fm_embedding_size=fm_embedding_size,
                input_layer_partitioner=input_layer_partitioner,
                config=config,
                batch_norm=batch_norm)

        super(DeepFM, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config,
            warm_start_from=warm_start_from)


def _DeepFM_model_fn(features,
                     labels,
                     mode,
                     head,
                     linear_feature_columns=None,
                     linear_optimizer='Ftrl',
                     linear_sparse_combiner='sum',
                     dnn_feature_columns=None,
                     dnn_optimizer='Adagrad',
                     dnn_hidden_units=None,
                     dnn_activation_fn=nn.relu,
                     dnn_dropout=None,
                     fm_embedding_size=None,
                     input_layer_partitioner=None,
                     config=None,
                     batch_norm=False):
    if not isinstance(features, dict):
        raise ValueError('features should be a dictionary of `Tensor`s. '
                         'Given type: {}'.format(type(features)))

    num_ps_replicas = config.num_ps_replicas if config else 0
    input_layer_partitioner = input_layer_partitioner or (
        partitioned_variables.min_max_variable_partitioner(
            max_partitions=num_ps_replicas,
            min_slice_size=64 << 20))

    _cols = [col.name for col in dnn_feature_columns]
    if not _cols:
        inputs = {}
    else:
        with variable_scope.variable_scope(
                "dnn_fm_inputs",
                partitioner=input_layer_partitioner):
            columns = list(set(dnn_feature_columns))
            inputs = myself_input_layer(features, columns)

    # FM_dnn logits
    dnn_fm_parent_scope = "dnn_fm"
    if not dnn_feature_columns:
        dnn_fm_logits = None
    else:
        optimizer = get_optimizer_instance(dnn_optimizer, None)
        _check_no_sync_replicas_optimizer(optimizer)
        with variable_scope.variable_scope(
                dnn_fm_parent_scope):
            dnn_fm_logit_fn = _dnn_fm_logit_fn_builder(
                units=head.logits_dimension,
                hidden_units=dnn_hidden_units,
                column_names=_cols,
                activation_fn=dnn_activation_fn,
                dropout=dnn_dropout,
                batch_norm=batch_norm,
                fm_embedding_size=fm_embedding_size)
            dnn_fm_logits = dnn_fm_logit_fn(inputs=inputs, mode=mode)

    # Linear logits
    linear_parent_scope = "linear"
    if not linear_feature_columns:
        linear_logits = None
    else:
        linear_optimizer = get_optimizer_instance(
            linear_optimizer, None)
        _check_no_sync_replicas_optimizer(linear_optimizer)
        with variable_scope.variable_scope(
                linear_parent_scope,
                values=tuple(six.itervalues(features)),
                partitioner=input_layer_partitioner) as scope:
            linear_logit_fn = _linear_logit_fn_builder(
                units=head.logits_dimension,
                feature_columns=linear_feature_columns,
                sparse_combiner=linear_sparse_combiner)
            linear_logits = linear_logit_fn(features=features)
            _add_layer_summary(linear_logits, scope.name)

    # Combine logits and build full model.
    if dnn_fm_logits is not None:
        ops.add_to_collection("deepFM_logits_collection", dnn_fm_logits)
    if linear_logits is not None:
        ops.add_to_collection("deepFM_logits_collection", linear_logits)

    logits = math_ops.add_n(ops.get_collection("deepFM_logits_collection"))

    def _train_op_fn(loss):
        train_ops = []
        global_step = training_util.get_global_step()

        if dnn_fm_logits is not None:
            train_ops.append(optimizer.minimize(
                loss,
                var_list=ops.get_collection(
                    ops.GraphKeys.TRAINABLE_VARIABLES,
                    scope=dnn_fm_parent_scope)))

        if linear_logits is not None:
            train_ops.append(linear_optimizer.minimize(
                loss,
                var_list=ops.get_collection(
                    ops.GraphKeys.TRAINABLE_VARIABLES,
                    scope=linear_parent_scope)))

        train_op = control_flow_ops.group(*train_ops)
        with ops.control_dependencies([train_op]):
            return distribute_lib.increment_var(global_step)

    return head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        train_op_fn=_train_op_fn,
        logits=logits)


def _linear_logit_fn_builder(units, feature_columns, sparse_combiner='sum'):
    def linear_logit_fn(features):
        cols_to_vars = {}
        logits = feature_column_lib.linear_model(
            features=features,
            feature_columns=feature_columns,
            units=units,
            sparse_combiner=sparse_combiner,
            cols_to_vars=cols_to_vars)
        bias = cols_to_vars.pop('bias')
        if units > 1:
            summary.histogram('bias', bias)
        else:
            # If units == 1, the bias value is a length-1 list of a scalar Tensor,
            # so we should provide a scalar summary.
            summary.scalar('bias', bias[0][0])
        summary.scalar('fraction_of_zero_weights',
                       _compute_fraction_of_zero(cols_to_vars))
        return logits

    return linear_logit_fn


def _dnn_fm_logit_fn_builder(units, hidden_units, column_names, activation_fn,
                             dropout, batch_norm, fm_embedding_size):
    if not isinstance(units, int):
        raise ValueError('units must be an int.  Given type: {}'.format(
            type(units)))

    def dnn_logit_fn(inputs, mode):
        is_training = mode == ModeKeys.TRAIN
        with variable_scope.variable_scope(
                'input_from_feature_columns'):
            dnn_inputs = []
            for c in column_names:
                dnn_inputs.append(inputs[c])
            net = array_ops.concat(dnn_inputs, axis=1)
        for layer_id, num_hidden_units in enumerate(hidden_units):
            with variable_scope.variable_scope(
                            'hiddenlayer_%d' % layer_id, values=(net,)) as hidden_layer_scope:
                net = core_layers.dense(
                    net,
                    units=num_hidden_units,
                    activation=activation_fn,
                    kernel_initializer=init_ops.glorot_uniform_initializer(),
                    name=hidden_layer_scope)
                if dropout is not None and is_training:
                    net = core_layers.dropout(net, rate=dropout, training=True)
                if batch_norm:
                    net = normalization.batch_normalization(
                        net,
                        momentum=0.999,
                        training=is_training,
                        name='batchnorm_%d' % layer_id)
            _add_hidden_layer_summary(net, hidden_layer_scope.name)

        with variable_scope.variable_scope('logits', values=(net,)) as logits_scope:
            logits = core_layers.dense(
                net,
                units=units,
                activation=None,
                kernel_initializer=init_ops.glorot_uniform_initializer(),
                name=logits_scope)
        _add_hidden_layer_summary(logits, logits_scope.name)
        return logits

    def fm_logit_fn(inputs):
        with variable_scope.variable_scope(
                'get_fm_inputs'):
            fm_filed_num = len(column_names)
            fm_inputs = []
            for c in column_names:
                fm_inputs.append(inputs[c])
            net = array_ops.concat(fm_inputs, axis=1)
            embeddings = gen_array_ops.reshape(net, (-1, fm_filed_num, fm_embedding_size))  # -1 * F * K
            # according to simplified formula
            summed_squared_emb = math_ops.square(math_ops.reduce_sum(embeddings, -2))  # -1 * K
            squared_summed_emb = math_ops.reduce_sum(math_ops.square(embeddings), -2)  # -1 * K
            logits = 0.5 * math_ops.reduce_sum(math_ops.subtract(summed_squared_emb, squared_summed_emb), -1)  # -1
            logits = array_ops.expand_dims(logits, -1)
            return logits

    def dnn_fm_logit_fn(inputs, mode):
        return fm_logit_fn(inputs) + dnn_logit_fn(inputs, mode)

    return dnn_fm_logit_fn


def _fm_logit_fn_builder(column_names, fm_embedding_size):
    def fm_logit_fn(inputs):
        with variable_scope.variable_scope(
                'get_fm_inputs'):
            fm_filed_num = len(column_names)
            fm_inputs = []
            for c in column_names:
                fm_inputs.append(inputs[c])
            net = array_ops.concat(fm_inputs, axis=1)
            embeddings = gen_array_ops.reshape(net, (-1, fm_filed_num, fm_embedding_size))  # -1 * F * K
            # according to simplified formula
            summed_squared_emb = math_ops.square(math_ops.reduce_sum(embeddings, -2))  # -1 * K
            squared_summed_emb = math_ops.reduce_sum(math_ops.square(embeddings), -2)  # -1 * K
            logits = 0.5 * math_ops.reduce_sum(math_ops.subtract(summed_squared_emb, squared_summed_emb), -1)  # -1
            logits = array_ops.expand_dims(logits, -1)
            return logits

    return fm_logit_fn


def myself_input_layer(features,
                       feature_columns,
                       weight_collections=None,
                       trainable=True,
                       cols_to_vars=None):
    feature_columns = _normalize_feature_columns(feature_columns)
    for column in feature_columns:
        if not isinstance(column, _DenseColumn):
            raise ValueError('Items of feature_columns must be a _DenseColumn. '
                             'You can wrap a categorical column with an '
                             'embedding_column or indicator_column. Given: {}'.format(column))
    weight_collections = list(weight_collections or [])
    if ops.GraphKeys.GLOBAL_VARIABLES not in weight_collections:
        weight_collections.append(ops.GraphKeys.GLOBAL_VARIABLES)
    if ops.GraphKeys.MODEL_VARIABLES not in weight_collections:
        weight_collections.append(ops.GraphKeys.MODEL_VARIABLES)

    # a non-None `scope` can allow for variable reuse, when, e.g., this function
    # is wrapped by a `make_template`.
    with variable_scope.variable_scope(None, default_name='myself_input_layer', values=features.values()):
        builder = _LazyBuilder(features)
        output_tensors = {}
        for column in feature_columns:
            with variable_scope.variable_scope(None, default_name=column._var_scope_name):
                tensor = column._get_dense_tensor(
                    builder,
                    weight_collections=weight_collections,
                    trainable=trainable)
                num_elements = column._variable_shape.num_elements()  # pylint: disable=protected-access
                batch_size = array_ops.shape(tensor)[0]
                output_tensors[column.name] = array_ops.reshape(tensor, shape=(batch_size, num_elements))
            if cols_to_vars is not None:
                # Retrieve any variables created (some _DenseColumn's don't create
                # variables, in which case an empty list is returned).
                cols_to_vars[column] = ops.get_collection(
                    ops.GraphKeys.GLOBAL_VARIABLES,
                    scope=variable_scope.get_variable_scope().name)
    return output_tensors
