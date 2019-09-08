#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/13 16:57
# @Author  : Yajun Yin
# @Note    :
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import MF_model
import time

_CSV_COLUMNS = ['user', 'item', 'rating']
_CSV_COLUMNS_DEFAULTS = [[0], [0], [0.0]]

flags = tf.app.flags
FLAGS = flags.FLAGS
timestamp = str(int(time.time()))
flags.DEFINE_string("model_dir", "./ckpt/" + timestamp, "ckpt")
flags.DEFINE_string("train", "./data/train.data", "train")
flags.DEFINE_string("test", "./data/validation.data", "validation")
flags.DEFINE_integer("num_users", 610, "num_users")
flags.DEFINE_integer("num_products", 9274, "num_products")
flags.DEFINE_string("model_name", "svd", "model_name")
flags.DEFINE_integer("num_epochs", 50, "num_epochs")
flags.DEFINE_integer("batch_size", 512, "batch_size")
flags.DEFINE_integer("embedding_size", 50, "embedding_size")
flags.DEFINE_integer("max_steps", 20000, "max_steps")
flags.DEFINE_integer("eval_steps", 100, "eval_steps")


def build_model_columns():
    user = tf.feature_column.categorical_column_with_identity("user", FLAGS.num_users + 1,
                                                              default_value=FLAGS.num_users)
    item = tf.feature_column.categorical_column_with_identity("item", FLAGS.num_products + 1,
                                                              default_value=FLAGS.num_products)

    # user embedding and item embedding
    user_emb = tf.feature_column.embedding_column(user, FLAGS.embedding_size,
                                                  initializer=tf.contrib.layers.xavier_initializer())
    item_emb = tf.feature_column.embedding_column(item, FLAGS.embedding_size,
                                                  initializer=tf.contrib.layers.xavier_initializer())
    # user bias and item bias
    user_bias = tf.feature_column.embedding_column(user, 1, initializer=tf.zeros_initializer())
    item_bias = tf.feature_column.embedding_column(item, 1, initializer=tf.zeros_initializer())
    return user_emb, item_emb, user_bias, item_bias


def input_fn(dataset, num_epochs, shuffle, batch_size):
    assert tf.gfile.Exists(dataset)

    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMNS_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('rating')
        return features, labels

    dataset = tf.data.TextLineDataset(dataset)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(parse_csv, num_parallel_calls=8).prefetch(40)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels


def build_estimator(model_name, columns):
    run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto())

    if model_name == 'svd':
        return MF_model.BaseMFModel(model_dir=FLAGS.model_dir,
                                    embedding_cols=[columns[0], columns[1]],
                                    bias_cols=[columns[2], columns[3]],
                                    reg_pen=0.0,
                                    learning_rate=2,
                                    is_implicit=True,
                                    loss='cross entropy',
                                    intercepts=True,
                                    optimizer="Ftrl",
                                    config=run_config)


def main(unused_argv):
    cols = build_model_columns()
    model = build_estimator(FLAGS.model_name, cols)

    def train_input_fn():
        return input_fn(FLAGS.train, 20, True, FLAGS.batch_size)

    def eval_input_fn():
        return input_fn(FLAGS.train, 1, False, FLAGS.batch_size)

    feature_spec = tf.feature_column.make_parse_example_spec(cols)
    hooks = []
    # export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    # exporter = tf.estimator.FinalExporter(FLAGS.model_name, export_input_fn)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=FLAGS.max_steps, hooks=hooks)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=FLAGS.eval_steps, start_delay_secs=30,
                                      throttle_secs=60, hooks=hooks)
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
