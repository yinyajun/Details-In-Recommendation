#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/9 11:34
# @Author  : Yajun Yin
# @Note    :

import tensorflow as tf

tf.set_random_seed(42)

input = tf.placeholder(tf.int64, shape=[None, 1], name="input")
label = tf.placeholder(tf.int64, shape=[None, 1], name="label")
embedding = tf.get_variable("embedding", [5, 3], dtype=tf.float32, trainable=True)
x = tf.nn.embedding_lookup(embedding, input)
#
with tf.variable_scope("deep"):
    logits1 = tf.layers.dense(x, 1, activation=None,
                              kernel_initializer=tf.ones_initializer())
    # print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="deep"))
with tf.variable_scope("fm"):
    logits2 = tf.reduce_sum(tf.square(x), -1)
    # print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="fm"))

with tf.variable_scope("loss"):
    logits1 = tf.reshape(logits1, [-1, 1])
    logits2 = tf.reshape(logits2, [-1, 1])
    logits = logits1 + logits2
    # loss
    labels = tf.to_float(label)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

    train_ops = []

    optimizer = tf.train.AdagradOptimizer(0.1)
    grads = optimizer.compute_gradients(loss)
    for grad in grads:
        print(grad[0])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    print(sess.run(embedding))
    print(sess.run([grad[0] for grad in grads], feed_dict={input: [[0], [1], [0]], label: [[1], [0], [1]]}))

    xx, l1, l2, lg, ls = sess.run([x, logits1, logits2, logits, loss],
                                  feed_dict={input: [[0], [1], [2]], label: [[1], [0], [1]]})
    print("xx:", xx)
    print("l1:", l1)
    print("l2:", l2)
    print("lg:", lg)
    print("ls:", ls)
