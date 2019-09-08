#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/9 19:20
# @Author  : Yajun Yin
# @Note    :

import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column.feature_column import _LazyBuilder
import numpy as np
from tensorflow.python.feature_column.feature_column import _normalize_feature_columns, \
    _verify_static_batch_size_equality, _LazyBuilder, _DenseColumn


def test_reuse():
    data = {'gender': [['M'], ['G'], ['M'], ['M']],
            'user': [['A'], ['B'], ['C'], ['C']],
            'pos': [['a'], ['d'], ['f'], ['c']],
            'neg': [['c'], ['e'], ['d'], ['a']]}
    user_v_list = ['A', 'B', 'C', 'D']
    item_v_list = ['a', 'b', 'c', 'd', 'e', 'f']

    gender_col = feature_column.categorical_column_with_vocabulary_list('gender', ['M', "G"], dtype=tf.string)
    user_col = feature_column.categorical_column_with_vocabulary_list('user', user_v_list, dtype=tf.string)
    pos_item_col = feature_column.categorical_column_with_vocabulary_list('pos', item_v_list, dtype=tf.string)
    neg_item_col = feature_column.categorical_column_with_vocabulary_list('neg', item_v_list, dtype=tf.string)

    gender_embedding = feature_column.embedding_column(gender_col, 2)
    user_embedding = feature_column.embedding_column(user_col, 2)
    pos_embedding, neg_embedding = feature_column.shared_embedding_columns(
        [pos_item_col, neg_item_col], 3)
    columns = [gender_embedding, user_embedding, pos_embedding, neg_embedding]

    with tf.variable_scope("a") as scope:
        aa = scope.name
        ret = tf.feature_column.input_layer(data, columns)
    print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=aa))

    with tf.variable_scope("b") as scope:
        bb = scope.name
        ret1 = tf.feature_column.input_layer(data, columns)
    print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=bb))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        print(sess.run(ret))
        print('------------------')
        print(sess.run(ret1))
        # r = []
        # for k, v in ret.items():
        #     print(k)
        #     print(sess.run(v))
        #     r.append(v)
        #
        # c = tf.concat(r, 1)
        # print(sess.run(c))
        # d = tf.reshape(c, [-1, 4, 4])
        # print(sess.run(d))


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
    if tf.GraphKeys.GLOBAL_VARIABLES not in weight_collections:
        weight_collections.append(tf.GraphKeys.GLOBAL_VARIABLES)
    if tf.GraphKeys.MODEL_VARIABLES not in weight_collections:
        weight_collections.append(tf.GraphKeys.MODEL_VARIABLES)

    # a non-None `scope` can allow for variable reuse, when, e.g., this function
    # is wrapped by a `make_template`.
    with tf.variable_scope(None, default_name='myself_input_layer', values=features.values()):
        builder = _LazyBuilder(features)
        output_tensors = {}
        for column in feature_columns:
            with tf.variable_scope(None, default_name=column._var_scope_name):
                tensor = column._get_dense_tensor(
                    builder,
                    weight_collections=weight_collections,
                    trainable=trainable)
                num_elements = column._variable_shape.num_elements()  # pylint: disable=protected-access
                batch_size = tf.shape(tensor)[0]
                output_tensors[column.name] = tf.reshape(tensor, shape=(batch_size, num_elements))
            if cols_to_vars is not None:
                # Retrieve any variables created (some _DenseColumn's don't create
                # variables, in which case an empty list is returned).
                cols_to_vars[column] = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES,
                    scope=tf.get_variable_scope().name)
    return output_tensors
