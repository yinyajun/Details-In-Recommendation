#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/10 13:25
# @Author  : Yajun Yin
# @Note    : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import math
import argparse
import tensorflow as tf
from util import *
import DeepCrossNetwork
from tensorflow.python.feature_column import feature_column as feature_column_lib

# default training set(source dataset refers to some private details. So I change it to public CENSUS DATASET )
"""
YOU CAN DOWNLOAD DATA FROM https://archive.ics.uci.edu/ml/machine-learning-databases/adult
PUT TRAIN DATA TO './data/train'
PUT EVALUATE DATA TO './data/test'
YOU CAN ALSO CONSULT OFFICIAL EXAMPLE IN https://github.com/tensorflow/models/blob/master/official/wide_deep/census_dataset.py
NOTE: I modify the following values and never test anymore, JUST TAKE IT AS AN EXAMPLE!
"""
_BATCH_SIZE_TRAIN = 100
_BATCH_SIZE_TEST = 100
_MODEL_DIR = './checkpoint/dcn'
_MODEL_TYPE = 'deep_cross'
_TRAIN_DATA = './data/train'
_TEST_DATA = './data/test'
_MAX_STEP = 4000
_EVAL_STEP = 10

# columns parse info
_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]
_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]


def init_parser():
    """add flags"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=_MODEL_DIR, help='Base directory for the model.')
    parser.add_argument('--model_type', type=str, default=_MODEL_TYPE, help="Valid model types: deep_cross.")
    parser.add_argument('--test_batch_size', type=int, default=_BATCH_SIZE_TEST,
                        help='Number of examples per batch when test.')
    parser.add_argument('--train_batch_size', type=int, default=_BATCH_SIZE_TRAIN,
                        help='Number of examples per batch when training.')
    parser.add_argument('--train_data', type=str, default=_TRAIN_DATA, help='Path to the training data.')
    parser.add_argument('--test_data', type=str, default=_TEST_DATA, help='Path to the test data.')
    parser.add_argument('--max_steps', type=int, default=_MAX_STEP, help='Number of training steps.')
    parser.add_argument('--eval_steps', type=int, default=_EVAL_STEP, help='Number of eval steps per evaluation.')
    return parser


def build_model_columns():
    """Builds a set of wide and deep feature columns."""
    # Continuous variable columns
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education', [
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
            'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
            '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', [
            'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship', [
            'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
            'Other-relative'])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass', [
            'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

    # To show an example of hashing:
    occupation = tf.feature_column.categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=_HASH_BUCKET_SIZE)

    columns = [
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        tf.feature_column.indicator_column(workclass),
        tf.feature_column.indicator_column(education),
        tf.feature_column.indicator_column(marital_status),
        tf.feature_column.indicator_column(relationship),
        # To show an example of embedding
        tf.feature_column.embedding_column(occupation, dimension=8),
    ]
    return columns


def build_estimator(model_dir, model_type, columns):
    """Build an estimator appropriate for the given model type."""
    sess_config = tf.ConfigProto()
    run_config = tf.estimator.RunConfig().replace(session_config=sess_config)

    if model_type == 'deep_cross':
        hidden_units = [32, 16, 8]
        return DeepCrossNetwork_v3_7.DeepCrossNetwork(model_dir=model_dir,
                                                      columns=columns,
                                                      dnn_hidden_units=hidden_units,
                                                      dnn_dropout=0.5,
                                                      weight_column='weight',
                                                      batch_norm=True,
                                                      config=run_config,
                                                      l2_reg=0.01,
                                                      optimizer=tf.train.AdamOptimizer,
                                                      optimizer_spec={'epsilon': 1e-4},
                                                      learning_rate_spec={'learning_rate': 0.001,
                                                                          'decay_method': 'cosine_decay',
                                                                          'decay_steps': 3000,
                                                                          'alpha': 0.50, }, )


def input_fn(data_dir, num_epochs, shuffle, batch_size):
    """Generate an input function for the Estimator."""
    assert tf.gfile.Exists(data_file), ('%s not found. Please make sure you have run census_dataset.py and '
                                        'set the --data_dir argument to the correct path.' % data_file)

    def parse_csv(value):
        tf.logging.info('Parsing {}'.format(data_file))
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('income_bracket')
        labels = tf.equal(labels, '>50K')  # binary classification
        return features, labels

    # Extract lines from input files using the Dataset API.
    data_files = [os.path.join(data_dir, f) for f in tf.gfile.ListDirectory(data_dir)
                  if tf.gfile.Exists(os.path.join(data_dir, f))]
    dataset = tf.data.TextLineDataset(data_files)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.map(parse_csv, num_parallel_calls=8).prefetch(4000)
    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def main(unused_argv):
    # get model
    columns = build_model_columns()
    model = build_estimator(FLAGS.model_dir, FLAGS.model_type, columns)

    def train_input_fn():
        return input_fn(FLAGS.train_data, 10, True, FLAGS.train_batch_size)

    def eval_input_fn():
        return input_fn(FLAGS.test_data, 1, False, FLAGS.test_batch_size)

    feature_spec = tf.feature_column.make_parse_example_spec(columns)
    hooks = []
    export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    exporter = tf.estimator.FinalExporter('deep_cross', export_input_fn)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=FLAGS.max_steps, hooks=hooks)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=FLAGS.eval_steps, start_delay_secs=300,
                                      throttle_secs=600, hooks=hooks, exporters=[exporter])
    # this api can implement both distributed and non-distributed training.
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = init_parser()
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
