#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/4 14:44
# @Author  : Yajun Yin
# @Note    :
import time
import os
import inference
import tensorflow as tf
import csv
import numpy as np

_CSV_COLUMNS = ['user', 'item', 'score', 'time']
_CSV_COLUMNS_DEFAULTS = [[0], [0], [0.0], [0]]

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("train", "./data/train.data", "train")
flags.DEFINE_string("test", "./data/validation.data", "validation")
flags.DEFINE_integer("seed", 566, "seed")
flags.DEFINE_integer("num_epochs", 20, "num_epochs")
flags.DEFINE_integer("num_steps", 200, "num_steps")
flags.DEFINE_string("logging_dir", "./tmp/tensorboard", "log_dir")
flags.DEFINE_string("model_name", "mf", "model_name")
flags.DEFINE_integer("batch_size", 2048, "batch_size")


def train(dataset):
    # placeholder
    user_list = tf.placeholder(tf.int32, [None], name="user_list_placeholder")
    prod_list = tf.placeholder(tf.int32, [None], name="product_list_placeholder")
    label_list = tf.placeholder(tf.float32, [None, 1], name="label_list_placeholder")

    scores, reg = inference.inference(user_list, prod_list)
    global_step = tf.Variable(0, trainable=False)

    # loss
    mse = tf.losses.mean_squared_error(labels=label_list, predictions=scores, reduction="weighted_mean")
    loss = mse + reg
    rmse = tf.sqrt(mse)
    print('rmse:', rmse)

    # opt
    train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss, global_step=global_step)

    tf.set_random_seed(FLAGS.seed)
    next_batch = dataset
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    saver = tf.train.Saver()
    t = time.time()
    with tf.Session(config=config) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        try:
            while True:
                user_batch, product_batch, label_batch = sess.run(next_batch)
                feed_dict = {user_list: user_batch,
                             prod_list: product_batch,
                             label_list: label_batch}
                _, loss_val, step = sess.run([train_op, loss, global_step], feed_dict=feed_dict)

                if step % FLAGS.num_steps == 0:
                    print("After %d steps, loss on training batch is %g. Time taken (s) = %s" % (
                        step, loss_val, str(round(time.time() - t))))
                    ####################################################
                    users_test, prods_test, labels_test = load_test(FLAGS.test)
                    test_feed_dict = {user_list: users_test,
                                      prod_list: prods_test,
                                      label_list: labels_test}
                    rmse_eval = sess.run(rmse, feed_dict=test_feed_dict)
                    t = time.time()
                    print("Test rmse at step %d is %s" % (step, rmse_eval))
                    print("############################################################################")

        except tf.errors.OutOfRangeError:
            print("Reached the number of epochs")
        finally:
            saver.save(sess, os.path.join(FLAGS.logging_dir, FLAGS.model_name), global_step=global_step)

        print("Training Finished!")


def main():
    dataset = input_fn(FLAGS.train, FLAGS.batch_size, FLAGS.num_epochs, shuffle=True)
    train(dataset)


def input_fn(data_dir, batch_size, num_epochs, shuffle):
    assert tf.gfile.Exists(data_dir)

    dataset = tf.data.TextLineDataset(data_dir).map(
        lambda line: tf.decode_csv(line, record_defaults=_CSV_COLUMNS_DEFAULTS), num_parallel_calls=8)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=50000)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()

    user_batch, product_batch, label_batch, _ = iterator.get_next()
    label_batch = tf.expand_dims(label_batch, -1)

    return user_batch, product_batch, label_batch


def load_test(data_dir):
    user_list = []
    product_list = []
    labels = []
    with open(data_dir, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            user_list.append(int(row[0]))
            product_list.append(int(row[1]))
            labels.append(float(row[2]))
    labels = np.reshape(labels, [-1, 1])
    return user_list, product_list, labels


if __name__ == '__main__':
    main()
