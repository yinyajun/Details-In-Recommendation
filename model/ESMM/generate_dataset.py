#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/9 19:38
# @Author  : Yajun Yin
# @Note    :

import numpy as np
import tensorflow as tf
import urllib
from urllib import request
import os

DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult'
TRAINING_FILE = 'adult.data'
TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)
EVAL_FILE = 'adult.test'
EVAL_URL = '%s/%s' % (DATA_URL, EVAL_FILE)


def _download_and_clean_file(filename, url):
    """Downloads data from url, and makes changes to match the CSV format."""
    temp_file, _ = urllib.request.urlretrieve(url, reporthook=_callbackfunc)
    with tf.gfile.Open(temp_file, 'r') as temp_eval_file:
        with tf.gfile.Open(filename, 'w') as eval_file:
            for line in temp_eval_file:
                line = line.strip()
                line = line.replace(', ', ',')
                if not line or ',' not in line:
                    continue
                if line[-1] == '.':
                    line = line[:-1]
                # add labels: is_click, is_convert
                is_click = np.random.binomial(1, 0.4)
                if is_click:
                    is_convert = np.random.binomial(1, 0.1)
                else:
                    is_convert = 0
                line = line + ',' + str(is_click) + ',' + str(is_convert)
                line += '\n'
                eval_file.write(line)
    tf.gfile.Remove(temp_file)


def download(data_dir):
    """Download census data if it is not already present."""
    tf.gfile.MakeDirs(data_dir)

    training_file_path = os.path.join(data_dir, TRAINING_FILE)
    if not tf.gfile.Exists(training_file_path):
        _download_and_clean_file(training_file_path, TRAINING_URL)

    eval_file_path = os.path.join(data_dir, EVAL_FILE)
    if not tf.gfile.Exists(eval_file_path):
        _download_and_clean_file(eval_file_path, EVAL_URL)


def _callbackfunc(blocknum, readsize, totalsize):
    """
    callback function of urllib.retrieve
    """
    if totalsize > 0:
        percentage = (blocknum + 1) * readsize / totalsize
    else:
        percentage = blocknum
    print("download : %d %%" % percentage)


if __name__ == '__main__':
    download("./data")
