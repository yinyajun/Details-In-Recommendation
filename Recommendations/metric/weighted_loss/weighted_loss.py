#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/27 10:16
# @Author  : Yajun Yin
# @Note    :
import pandas as pd
from dcn import *  # Recommendations/model/DeepCrossNetwork/dcn.py
import numpy as np

# RANKING_DATA is generated in generate_ranking_test.py
RANKING_DATA = "hdfs://~/20180820/ranking"
MODEL_TYPE = "deep_cross"
# Your model checkpoint
CKPT_DIR = "hdfs://~/yinyajun/model/dcn_non_real3"

BATCH = 1000

# These apis from dcn.py, it generate an tf.estimator to predict examples
columns = build_model_columns()
model = build_estimator(CKPT_DIR, MODEL_TYPE, columns)

# This api from dcn.py, it is tf.data.TextLineDataset to read RANKING_DATA
def eval_input_fn():
    return input_fn(RANKING_DATA, 1, False, BATCH)


def weighted_loss(labels, weights, preds):
    print("calc per_user_weighted_loss...")
    preds = np.array(preds)
    labels = np.array(labels)
    weights = np.array(weights)
    num_of_1 = np.sum(labels == 1)
    index = np.argsort(-labels)

    positive_label_index = index[:num_of_1]
    negative_label_index = index[num_of_1:]

    mis_predict_time = 0
    total_predict_time = 0
	# construct (pos, neg) pair
    for i in positive_label_index:
        for j in negative_label_index:
            total_predict_time += weights[i]  # the watch time value of negative example is zero
            if preds[i] <= preds[j]:  # negative example got higher score
                mis_predict_time += weights[i]
    try:
        weighted_loss = mis_predict_time / total_predict_time  # divisor cannot be zero
    except ZeroDivisionError:
        weighted_loss = 0
    return weighted_loss


# get preds
preds = model.predict(eval_input_fn, predict_keys="probabilities")
preds = (pred["probabilities"][1] for pred in preds)

print("Begin to calc weighted loss...")
feature, label = input_fn(RANKING_DATA, 1, False, BATCH)
labels = []
weights = []
uids = []
with tf.Session() as sess:
    while True:
        try:
            uid = feature["uid"]
            weight = feature["weight"]
            uu, ll, ww = sess.run([uid, label, weight])
            labels.extend(ll)
            uids.extend(uu)
            weights.extend(ww)
        except tf.errors.OutOfRangeError:
            break

data = {"uid": uids,
        "label": labels,
        "pred": list(preds),
        "weight": weights}

df = pd.DataFrame(data)
per_user_weighted_loss = df.groupby('uid').apply(lambda x: weighted_loss(x.label, x.weight, x.pred))
mean_weighted_loss = np.sum(per_user_weighted_loss) / len(per_user_weighted_loss)
print("mean_weighted_loss:%f" % float(mean_weighted_loss))
