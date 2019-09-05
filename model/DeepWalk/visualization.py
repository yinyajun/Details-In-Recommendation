#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/5 9:50
# @Author  : Yajun Yin
# @Note    :

import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def load_embedding_from_file(file):
    nodes = []
    vectors = []
    with open(file, 'r') as f:
        for line in f:
            cols = line.strip().split()
            node = cols[0]
            vector = cols[1:]
            nodes.append(node)
            vectors.append(vector)
    return nodes, vectors


def load_node_label_from_file(file):
    nodes = []
    labels = []
    with open(file, 'r') as f:
        for line in f:
            cols = line.strip().split()
            node = cols[0]
            label = cols[1]
            nodes.append(node)
            labels.append(label)
    return nodes, labels


def plot_embeddings(emb_file, ):
    nodes, embeddings = load_embedding_from_file(emb_file)
    embeddings = np.array(embeddings)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(embeddings)

    color_idx = {}
    for i in range(len(nodes)):
        color_idx[nodes[i]] = i

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()
