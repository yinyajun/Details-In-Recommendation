#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/12 17:48
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import abc
import numpy as np
import time
import utils as ut
from pyspark import SparkConf, SparkContext
from pyspark.sql import HiveContext


conf = SparkConf()
conf.set("spark.hadoop.validateOutputSpecs", "false")
conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
conf.set("spark.shuffle.spill", "true")
conf.set("spark.speculation", "true")
sc = SparkContext(conf=conf)
hc = HiveContext(sc)


def ReadRatings_GlobalSplit(ratingFileDir, testRatio):
    userCount = 0
    itemCount = 0
    print("Global splitting with testRatio " + str(testRatio))

    # Step 1. Construct data structure for sorting.
    print("Read ratings and sort.")
    startTime  = time.time()
    ratings = sc.textFile(ratingFileDir)

    # Step 2. Sort the ratings by time (small->large).

    # Step 3. Generate trainMatrix and testStream




def ReadRatings_HoldOneOut(ratingFile):
    pass

def deduplicate(ratingFile):
    pass

def evaluate_model(model, name):
    pass

def evaluate_model_online(model, name, interval):
    pass




def main():
    pass


if __name__ == '__main__':



