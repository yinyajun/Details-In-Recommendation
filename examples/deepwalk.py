#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/7 12:58
# @Author  : Yajun Yin
# @Note    :

from models.DeepWalk import DeepWalk


def get_input_rdd():
    # ...
    return rdd


def train(dataset, path):
    m = DeepWalk(num_paths=10)
    vectors = m.fit(dataset)
    m.save_vectors(path=path, vectors=vectors)


if __name__ == '__main__':
    rdd = get_input_rdd()  # format: [user, item, timestamp]
    file = "/tmp/random_walk.emb"

    train(rdd, file)

# 2019-09-08 21:08:38,298  deep_walk.model  INFO
# alpha: Restart Probability (default: 0)
# learning_rate: Skip gram learning rate (default: 0.025)
# max_active_num: Avoid Spam User, max impressed items num per user (default: 200)
# min_count: Min count of a vertex in corpus (default: 50)
# num_iter: Skip gram iteration nums (default: 1)
# num_partitions: Num partitions of training skip gram (default: 10)
# num_paths: The nums of paths for a vertex (default: 20)
# num_workers: Parallelism for generation walks (default: 400)
# path_length: The maximum length of a path (default: 50)
# pre_process: Whether use processing function or not (default: True)
# session_duration: The duration for session segment (default: 3600)
# vector_size: Low latent vector size (default: 50)
# 2019-09-08 21:08:53,707  deep_walk.model  INFO  Create graph...
# 2019-09-08 21:10:49,455  deep_walk.model  INFO  Create graph done.
# 2019-09-08 21:10:49,455  deep_walk.model  INFO  Generate walks...
# 2019-09-08 21:11:05,129  deep_walk.model  INFO  walks rdd num: 2098500
# 2019-09-08 21:11:05,129  deep_walk.model  INFO  Generate walks done.
# 2019-09-08 21:11:05,129  deep_walk.model  INFO  Skip gram...
# 2019-09-08 21:41:05,129  deep_walk.model  INFO  Skip gram done.
