# coding=UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import model

_CSV_COLUMNS = ['user', 'item', 'rating']
_CSV_COLUMNS_DEFAULTS = [[0], [0], [0]]

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("model_dir", "./ckpt", "ckpt")
flags.DEFINE_string("train", "./Data/train.csv", "train")
flags.DEFINE_string("test", "./Data/train.csv", "test")
flags.DEFINE_integer("num_users", 610, "num_users")
flags.DEFINE_integer("num_products", 9274, "num_products")
flags.DEFINE_string("model_name", "bias_svd", "model_name")
flags.DEFINE_integer("num_epochs", 30, "num_epochs")
flags.DEFINE_integer("batch_size", 512, "batch_size")
flags.DEFINE_integer("embedding_size", 50, "embedding_size")
flags.DEFINE_integer("max_steps", 6000, "max_steps")
flags.DEFINE_integer("eval_steps", 100, "eval_steps")


def build_model_columns():
    user = tf.feature_column.categorical_column_with_identity("user", FLAGS.num_users + 1,
                                                              default_value=FLAGS.num_users)
    item = tf.feature_column.categorical_column_with_identity("item", FLAGS.num_products + 1,
                                                              default_value=FLAGS.num_products)
    user_emb = tf.feature_column.embedding_column(user, FLAGS.embedding_size,
                                                  initializer=tf.contrib.layers.xavier_initializer())
    item_emb = tf.feature_column.embedding_column(item, FLAGS.embedding_size,
                                                  initializer=tf.contrib.layers.xavier_initializer())
    user_bias = tf.feature_column.embedding_column(user, 1, initializer=tf.zeros_initializer())
    item_bias = tf.feature_column.embedding_column(item, 1, initializer=tf.zeros_initializer())
    return user_emb, item_emb, user_bias, item_bias


def input_fn(dataset, num_epochs, shuffle, batch_size):
    assert tf.gfile.Exists(dataset)

    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMNS_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('rating')
        # labels = tf.expand_dims(labels, axis=-1)
        return features, labels

    dataset = tf.data.TextLineDataset(dataset)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(parse_csv, num_parallel_calls=8).prefetch(40)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    # print(labels)
    return features, labels


def build_estimator(model_name, cols):
    run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto())

    if model_name == 'svd':
        return model.Svd(model_dir=FLAGS.model_dir,
                         model_name=FLAGS.model_name,
                         user_emb=cols[0],
                         item_emb=cols[1],
                         l2_pen=0.0,
                         learning_rate=1,
                         config=run_config)
    if model_name == "bias_svd":
        return model.BiasedSvd(model_dir=FLAGS.model_dir,
                               model_name=FLAGS.model_name,
                               user_emb=cols[0],
                               item_emb=cols[1],
                               user_bias=cols[2],
                               item_bias=cols[3],
                               l2_pen=0.0,
                               learning_rate=1,
                               config=run_config)


def main(unused_argv):
    cols = build_model_columns()
    model = build_estimator(FLAGS.model_name, cols)

    def train_input_fn():
        return input_fn(FLAGS.train, 20, True, FLAGS.batch_size)

    def eval_input_fn():
        return input_fn(FLAGS.test, 1, False, FLAGS.batch_size)

    feature_spec = tf.feature_column.make_parse_example_spec(cols)
    hooks = []
    export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    exporter = tf.estimator.FinalExporter(FLAGS.model_name, export_input_fn)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=FLAGS.max_steps, hooks=hooks)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=FLAGS.eval_steps, start_delay_secs=30,
                                      throttle_secs=60, hooks=hooks, exporters=[exporter])
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
