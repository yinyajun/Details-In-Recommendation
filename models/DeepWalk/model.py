#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/7 12:58
# @Author  : Yajun Yin
# @Note    :


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import copy

from pyspark.ml import Estimator
from pyspark import SparkContext
from pyspark.ml.param import Param, Params
from pyspark.mllib.feature import Word2Vec

from .walks import generate_walks
from .graph import load_adjacency_list_from_spark
from .preprocess import preprocessing


# TypeConverters is not supported in Spark 1.6.0
# window_size is not supported in Word2Vec in Spark 1.6.0
class DeepWalk(Estimator):
    # pre_process
    pre_process = Param(Params._dummy(),
                        "pre_process",
                        "Whether use processing function or not")
    max_active_num = Param(Params._dummy(),
                           "max_active_num",
                           "Avoid Spam User, max impressed items num per user")
    session_duration = Param(Params._dummy(),
                             "session_duration",
                             "The duration for session segment")
    # walkers
    num_workers = Param(Params._dummy(),
                        "num_workers",
                        "Parallelism for generation walks")
    num_paths = Param(Params._dummy(),
                      "num_paths",
                      "The nums of paths for a vertex")
    path_length = Param(Params._dummy(),
                        "path_length",
                        "The maximum length of a path")
    alpha = Param(Params._dummy(),
                  "alpha",
                  "Restart Probability")
    # word2vec
    vector_size = Param(Params._dummy(),
                        "vector_size",
                        "Low latent vector size")
    min_count = Param(Params._dummy(),
                      "min_count",
                      "Min count of a vertex in corpus")
    num_partitions = Param(Params._dummy(),
                           "num_partitions",
                           "Num partitions of training skip gram")
    num_iter = Param(Params._dummy(),
                     "num_iter",
                     "Skip gram iteration nums")
    learning_rate = Param(Params._dummy(),
                          "learning_rate",
                          "Skip gram learning rate")

    def __init__(self, **kwargs):
        super(DeepWalk, self).__init__()
        self._copy_params()
        self._init_logger()
        self._setDefault(pre_process=True,
                         max_active_num=200,
                         session_duration=3600,
                         num_workers=400,
                         num_paths=20,
                         path_length=50,
                         alpha=0,
                         vector_size=50,
                         min_count=50,
                         num_partitions=10,
                         num_iter=1,
                         learning_rate=0.025)
        self.logger.info("\n" + self.explainParams())
        self.set_params(**kwargs)

    def _copy_params(self):
        # this function is not implemented in Spark1.6.0
        cls = type(self)
        src_name_attrs = [(x, getattr(cls, x)) for x in dir(cls)]
        src_params = list(filter(lambda nameAttr: isinstance(nameAttr[1], Param), src_name_attrs))
        for name, param in src_params:
            setattr(self, name, copy_new_parent(param, self))

    def _init_logger(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)s  %(levelname)s  %(message)s')
        self.logger = logging.getLogger(__name__)

    def set_params(self, **kwargs):
        return self._set(**kwargs)

    def _fit(self, dataset):
        self.logger.info("Create graph...")
        rdd = self._pre_processing(dataset)
        self._create_graph(rdd)
        self.logger.info("Create graph done.")
        self.logger.info("Generate walks...")
        walks_rdd = self._generate_walks()
        self.logger.info("walks rdd num: {cnt}".format(cnt=walks_rdd.count()))
        self.logger.info("Generate walks done.")
        self.logger.info("Skip gram...")
        vectors = self._skip_gram(walks_rdd)
        self.logger.info("Skip gram done.")
        return vectors

    def _create_graph(self, rdd):
        self.G = load_adjacency_list_from_spark(rdd)

    def _generate_walks(self):
        num_paths = self.getOrDefault("num_paths")
        path_length = self.getOrDefault("path_length")
        num_workers = self.getOrDefault("num_workers")
        alpha = self.getOrDefault("alpha")
        sc = SparkContext._active_spark_context
        walks_rdd = generate_walks(sc, self.G, num_paths, path_length, num_workers, alpha)
        return walks_rdd

    def _pre_processing(self, dataset):
        # set pre_process False to input custom source RDD
        rdd = dataset
        if self.getOrDefault("pre_process"):
            max_active_num = self.getOrDefault("max_active_num")
            session_duration = self.getOrDefault("session_duration")
            rdd = preprocessing(rdd, max_active_num, session_duration)
        return rdd

    def _skip_gram(self, walks_rdd):
        vector_size = self.getOrDefault("vector_size")
        min_count = self.getOrDefault("min_count")
        num_partitions = self.getOrDefault("num_partitions")
        learning_rate = self.getOrDefault("learning_rate")
        num_iter = self.getOrDefault("num_iter")
        model = Word2Vec() \
            .setVectorSize(vector_size) \
            .setMinCount(min_count) \
            .setNumPartitions(num_partitions) \
            .setLearningRate(learning_rate) \
            .setNumIterations(num_iter) \
            .fit(walks_rdd)
        return model.getVectors()

    @staticmethod
    def save_vectors(path, vectors):
        with open(path, 'w') as f:
            for item, vec in vectors.items():
                f.write("{item} {vec}\n".format(item=str(item), vec=' '.join([str(v) for v in vec])))


def copy_new_parent(param, parent):
    # this function is not implemented in Spark1.6.0
    assert isinstance(param, Param)
    if param.parent == "undefined":
        new_param = copy.copy(param)
        new_param.parent = parent.uid
        return new_param
    else:
        raise ValueError("Cannot copy from non-dummy parent %s." % parent)
