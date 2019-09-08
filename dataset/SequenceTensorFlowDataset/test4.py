#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/20 14:18
# @Author  : Yajun Yin
# @Note    :

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
from tensorflow.python.estimator.export.export import build_parsing_serving_input_receiver_fn
from tensorflow.python.feature_column import feature_column as fc
from tensorflow.python.feature_column.feature_column import *
import json
import os
import math
from datetime import datetime

_BATCH_SIZE = 2


def initParser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir', type=str, default='./checkpoint',
        help='Base directory for the model.')
    parser.add_argument(
        '--model_type', type=str, default='wide_deep',
        help="Valid model types: {'wide', 'deep', 'wide_deep'}.")
    parser.add_argument(
        '--train_epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument(
        '--epochs_per_eval', type=int, default=1,
        help='The number of training epochs to run between evaluations.')
    parser.add_argument(
        '--batch_size', type=int, default=_BATCH_SIZE, help='Number of examples per batch.')
    parser.add_argument(
        '--train_data', type=str, default='./records2',
        help='Path to the training data.')
    parser.add_argument(
        '--test_data', type=str, default='./records3',
        help='Path to the test data.')
    return parser


def build_model_columns():
    week_list = fc.categorical_column_with_vocabulary_list("week_list",
                                                           vocabulary_list=['mon', 'tue', 'wed', 'thur', 'fri', 'sat',
                                                                            'sun'])
    week = fc.weighted_categorical_column(week_list, 'week_weight')

    week = fc.embedding_column(week, 3)

    wide = []
    deep = [week]
    return wide, deep


def build_estimator(model_dir, model_type, wide_columns, deep_columns):
    """Build an estimator appropriate for the given model type."""
    hidden_units = [16, 8, 4]

    # opt = tf.train.AdamOptimizer()
    opt = tf.train.ProximalAdagradOptimizer(learning_rate=0.01,
                                            l1_regularization_strength=0.01,
                                            l2_regularization_strength=0.01)
    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto())
    if model_type == 'wide_deep':
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=hidden_units,
            linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.01,
                                                    l1_regularization_strength=0.01,
                                                    l2_regularization_strength=0.01),
            dnn_optimizer=opt,
            config=run_config,
            dnn_dropout=0.5)


def input_fn(data_file, batch_size):
    """Generate an input function for the Estimator."""
    assert tf.gfile.Exists(data_file), (
            '%s not found. Please make sure you have set both arguments --train_data and --test_data.' % data_file)

    def parse(serialized_example):
        context_feature = {'label': tf.FixedLenFeature([], dtype=tf.int64)}
        sequence_features = {
            "week_list": tf.FixedLenSequenceFeature([], dtype=tf.string),
            'week_weight': tf.FixedLenSequenceFeature([], dtype=tf.float32)}
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=serialized_example,
            context_features=context_feature,
            sequence_features=sequence_features)
        labels = context_parsed['label']
        week_list = sequence_parsed['week_list']
        week_weight = sequence_parsed['week_weight']
        return labels, week_list, week_weight

    def form_features(*line):
        cols = ['label', 'week_list', 'week_weight']
        features = dict(zip(cols, line))
        label = features.pop('label')
        return features, label

    dataset = tf.data.TFRecordDataset(data_file) \
        .map(parse) \
        .padded_batch(batch_size, ([1], [7], [7])) \
        .map(form_features)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def main(unused_argv):
    wide_columns, deep_columns = build_model_columns()
    model = build_estimator(
        FLAGS.model_dir, FLAGS.model_type, wide_columns, deep_columns)

    def train_input_fn():
        return input_fn(
            FLAGS.train_data,
            2)

    def eval_input_fn():
        return input_fn(
            FLAGS.test_data,
            1)

    feature_spec = make_parse_example_spec(wide_columns + deep_columns)
    # export_input_fn = build_parsing_serving_input_receiver_fn(feature_spec)
    # exporter = tf.estimator.FinalExporter('realtime_reward', export_input_fn)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=50)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=4, throttle_secs=10)
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = initParser()
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)