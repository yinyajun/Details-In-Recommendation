#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/27 18:46

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import tensorflow as tf


# from train import *


def _mrr(true_item, ranklist):
    if true_item in ranklist:
        index = np.where(ranklist == true_item)[0][0]
        position = index + 1
        return np.reciprocal(float(position))
    else:
        return 0


def _hit(true_item, ranklist):
    return 1 if true_item in ranklist else 0


def _ndcg(true_item, ranklist):
    """ NDCG under this special scenario:
    * only one positive item (relevant item).
    * ranking score r_j can be only 1 or 0, which means Gain 2^(r_j) -1 can be 1 or 0.
    * ideal ranking is [1,0,0,0,...], so IDCG must be 1.
    * actual ranking may be [0,0,1,0,0,...]
    NDCG: Normalization, Cumulating, Gain, Position Discount
    more details in: https://www.cnblogs.com/eyeszjwang/articles/2368087.html
    DCG = 1/log(1+pos), pos is the position of relevant item
    NDCG = DCG/IDCG = DCG, since IDCG =1
    so, NDCG = 1/log(1+pos)
    """
    if true_item in ranklist:
        # since only 1 positive item
        index = np.where(ranklist == true_item)[0][0]
        position = index + 1
        return np.reciprocal(np.log2(position + 1))  # index start from 0
    return 0


def evaluation_dataset(dataset_location):
    record_defaults = [[1], [1], [0.]]
    dataset = tf.data.TextLineDataset(dataset_location) \
        .map(lambda line: tf.decode_csv(line, record_defaults=record_defaults))
    dataset = dataset.batch(100)
    iterator = dataset.make_one_shot_iterator()
    user_batch, product_batch, label_batch = iterator.get_next()
    label_batch = tf.expand_dims(label_batch, 1)
    return user_batch, product_batch, label_batch


def evaluation_model(sess, model):
    next_batch = evaluation_dataset("data/evaluate.csv")
    HR, MRR, NDCG = [], [], []
    try:
        while True:
            user_batch, product_batch, label_batch = sess.run(next_batch)
            feed_dict = {model.user_list: user_batch,
                         model.prod_list: product_batch,
                         model.label_list: label_batch,
                         model.treatment_prod_list: product_batch}
            pred = sess.run(model.prediction, feed_dict=feed_dict)
            pred = np.asarray(pred).flatten()
            # argsort returns the indices that would sort an array
            preds_index = np.argsort(- pred, kind='quicksort')  # sort by desc
            top_k_ranklist = preds_index[:10]  # @10
            NDCG.append(_ndcg(0, top_k_ranklist))
            HR.append(_hit(0, top_k_ranklist))
            MRR.append(_mrr(0, top_k_ranklist))
    except tf.errors.OutOfRangeError:
        print('evaluation over')
    hr = np.array(HR).mean()
    mrr = np.array(MRR).mean()
    ndcg = np.array(NDCG).mean()
    print("ndcg:", ndcg)
    print("hr:", hr)
    print("mrr:", mrr)


