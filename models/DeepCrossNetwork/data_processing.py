#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import urllib
import os

DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult'
TRAINING_FILE = 'adult.data'
TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)
EVAL_FILE = 'adult.test'
EVAL_URL = '%s/%s' % (DATA_URL, EVAL_FILE)


def _download_and_clean_file(filename, url):
    """Downloads data from url, and makes changes to match the CSV format."""
    temp_file, _ = urllib.request.urlretrieve(url)
    with tf.gfile.Open(temp_file, 'r') as temp_eval_file:
        with tf.gfile.Open(filename, 'w') as eval_file:
            for line in temp_eval_file:
                line = line.strip()
                line = line.replace(', ', ',')
                if not line or ',' not in line:
                    continue
                if line[-1] == '.':
                    line = line[:-1]
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


if __name__ == '__main__':
    download('./dataset')
