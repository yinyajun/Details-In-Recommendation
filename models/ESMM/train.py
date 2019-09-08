#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/9 18:42
# @Author  : Yajun Yin
# @Note    :

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
from tensorflow.python.estimator.export.export import build_parsing_serving_input_receiver_fn
from tensorflow.python.feature_column.feature_column import *
from ESMM_wide_deep import *
from ESMM import *

_BATCH_SIZE_TRAIN = 100
_BATCH_SIZE_TEST = 50
_MODEL_DIR = './checkpoint/esmm'
_MODEL_TYPE = 'esmm_wd'
_TRAIN_DATA = './data/adult.data'
_TEST_DATA = './data/adult.test'
_MAX_STEP = 6000
_EVAL_STEP = 60

# columns parse info
_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket', 'is_click', 'is_convert'
]
_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], [''], [0], [0]]
_HASH_BUCKET_SIZE = 1000


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

    income_bracket = tf.feature_column.categorical_column_with_vocabulary_list(
        "income_bracket", ["<=50K", ">50k"])

    # cross
    workclass_education = tf.feature_column.crossed_column(["workclass", "education"], 128)

    wide_columns = [workclass_education]

    deep_columns = [
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        tf.feature_column.indicator_column(workclass),
        tf.feature_column.indicator_column(education),
        tf.feature_column.indicator_column(marital_status),
        tf.feature_column.indicator_column(relationship),
        tf.feature_column.indicator_column(income_bracket),
        # To show an example of embedding
        tf.feature_column.embedding_column(occupation, dimension=8),
    ]
    return wide_columns, deep_columns


def build_estimator(model_dir, model_type, wide_columns, deep_columns):
    """Build an estimator appropriate for the given model type."""
    hidden_units = [64, 64, 64]
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}))

    if model_type == 'esmm':
        return ESMM(model_dir=model_dir,
                    columns=deep_columns,
                    dnn_hidden_units=hidden_units,
                    dnn_dropout=0.5,
                    config=run_config,
                    optimizer=tf.train.AdamOptimizer(learning_rate=0.001))
    elif model_type == 'esmm_wd':
        return ESMM_W_D(model_dir=model_dir,
                        linear_feature_columns=wide_columns,
                        dnn_feature_columns=deep_columns,
                        linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.01,
                                                                l1_regularization_strength=0.01,
                                                                l2_regularization_strength=0.01),
                        ctr_weight_column=None,
                        ctcvr_weight_column=None,
                        dnn_optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.01,
                                                                        l1_regularization_strength=0.01,
                                                                        l2_regularization_strength=0.01),
                        dnn_hidden_units=hidden_units,
                        dnn_dropout=0.5,
                        config=run_config)


def input_fn(data_file, num_epochs, shuffle, batch_size):
    """Generate an input function for the Estimator."""
    assert tf.gfile.Exists(data_file), ('%s not found. Please make sure you have run generate_dataset.py and '
                                        'set the --data_dir argument to the correct path.' % data_file)

    def parse_csv(value):
        tf.logging.info('Parsing {}'.format(data_file))
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        click_label = features.pop("is_click")
        convert_label = features.pop("is_convert")
        labels = {"click_label": click_label, "convert_label": convert_label}
        return features, labels

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)
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
    wide_columns, deep_columns = build_model_columns()
    model = build_estimator(FLAGS.model_dir, FLAGS.model_type, wide_columns, deep_columns)

    def train_input_fn():
        return input_fn(
            FLAGS.train_data,
            100,
            True,
            FLAGS.train_batch_size)

    def eval_input_fn():
        return input_fn(
            FLAGS.test_data,
            1,
            False,
            FLAGS.test_batch_size)

    feature_spec = make_parse_example_spec(wide_columns + deep_columns)
    export_input_fn = build_parsing_serving_input_receiver_fn(feature_spec)
    exporter = tf.estimator.FinalExporter('esmm', export_input_fn)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=FLAGS.max_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=FLAGS.eval_steps, start_delay_secs=60,
                                      throttle_secs=60, exporters=[exporter])
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = init_parser()
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
