#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/7 12:58
# @Author  : Yajun Yin
# @Note    :


from model.DeepWalk import DeepWalk


def get_input_rdd():
    # ...
    return rdd


def main(dataset, path):
    m = DeepWalk(num_paths=50)
    vectors = m.fit(dataset)
    m.save_vectors(path=path, vectors=vectors)


if __name__ == '__main__':
    rdd = get_source_rdd()
    path = "/tmp/random_walk.emb"

    main(rdd, path)
