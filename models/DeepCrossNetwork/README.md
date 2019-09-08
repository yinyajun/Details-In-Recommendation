# Deep Cross Network
* DCN is implemented using customized tf.estimator. This high-level api in tensorflow is very neat.
* Most of the feature engineering is conducted by tf.feature_column api.
* Use tf.dataset api to read data.
* mimic ``tf.estimator.DNNLinearCombinedClassifier``. 
-----
# Code Organization
* ``DeepCrossNetwork.py``: DCN model, subclass of tf.estimator.
* ``train.py``: Train script.
* ``data_processing.py``: Download dataset and process it.
-----
# Train
1. `python data_processing.py` to generate census dataset.
2. `python train.py` to run DCN on census dataset.
3.  ALL CLI arguments have default value, pls consult `train.py`.
-----
# Evaluation
*parameters not yet tuned for this toy example.*

INFO:tensorflow:Saving dict for global step 1000:

accuracy = 0.8217448,

accuracy_baseline = 0.76601565,

auc = 0.83714986,

average_loss = 3.32518,

global_step = 1000,

loss = 3.32518,

precision = 0.73568285,

recall = 0.37173066
