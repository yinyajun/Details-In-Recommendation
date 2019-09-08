#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf


def load_test_ratings(fname):
    def process_line(line):
        tmp = map(int, line.split('\t')[0:2])
        return list(tmp)

    ratings = map(process_line, open(fname, 'r'))
    return list(ratings)


def load_test_negs(fname):
    def process_line(line):
        tmp = map(int, line.split('\t'))
        return list(tmp)

    negs = map(process_line, open(fname, 'r'))
    return list(negs)


def train_input_fn(file, batch_size, num_epochs, shuffle):
    assert tf.gfile.Exists(file)

    _CSV_COLUMNS_DEFAULTS = [[0], [0], [0]]
    dataset = tf.data.TextLineDataset(file).map(
        lambda line: tf.decode_csv(line, record_defaults=_CSV_COLUMNS_DEFAULTS, field_delim='\t'), num_parallel_calls=8)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=50000)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    user_batch, item_batch, rating_batch = iterator.get_next()
    return user_batch, item_batch, rating_batch
