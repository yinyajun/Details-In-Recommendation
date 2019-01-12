#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/4 12:08
# @Author  : Yajun Yin
# @Note    :


import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("num_users", 69878, "num_users")
flags.DEFINE_integer("num_products", 10677, "num_products")
flags.DEFINE_integer("embedding_size", 16, "embedding_size")
flags.DEFINE_float("l2_pen", 0.1, "l2 penalty")
flags.DEFINE_float("learning_rate", 0.1, "learning_rate")

num_users = FLAGS.num_users
num_products = FLAGS.num_products
embedding_dim = FLAGS.embedding_size
l2_pen = FLAGS.l2_pen
learning_rate = FLAGS.learning_rate


def inference(user_batch, prod_batch):
    with tf.variable_scope('embedding'):
        user_embeddings = tf.get_variable("user_embeddings", shape=[num_users, embedding_dim],
                                          initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
        user_b = tf.Variable(tf.zeros([num_users]), name='user_b', trainable=True)
        product_embeddings = tf.get_variable("product_embeddings", shape=[num_products, embedding_dim],
                                             initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
        prod_b = tf.Variable(tf.zeros([num_products]), name='prod_b', trainable=True)

        user_embed = tf.nn.embedding_lookup(user_embeddings, user_batch)
        user_bias_embed = tf.nn.embedding_lookup(user_b, user_batch)
        product_embed = tf.nn.embedding_lookup(product_embeddings, prod_batch)
        prod_bias_embed = tf.nn.embedding_lookup(prod_b, prod_batch)

    with tf.variable_scope('logits'):
        global_bias = tf.get_variable('global_bias', [1], initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                                      trainable=True)
        scores = tf.reduce_sum(tf.multiply(user_embed, product_embed), 1)
        bias = tf.add(user_bias_embed, prod_bias_embed) + global_bias
        logits = scores + bias
        logits = tf.expand_dims(logits, -1)

    with tf.variable_scope('reg'):
        reg_vec = tf.nn.l2_loss(user_embed) + tf.nn.l2_loss(product_embed)
        reg_bias = tf.nn.l2_loss(prod_bias_embed) + tf.nn.l2_loss(user_bias_embed)
        reg_term = l2_pen * (reg_vec + reg_bias)

    return logits, reg_term
