# ESMM
 * Implemented using customized tf.estimator
 * Multi-task learning with two labels: click label and convert label. Click->view an item, convert->after view and buy this item. 
-----
# Code Organization
* ``ESMM.py``: ESMM model whose base model is dnn.
* ``ESMM_wide_deep``: ESMM model whose base model is wide&deep.
* ``train.py``: Train.
* ``generate_dataset.py``: Download census dataset and add labels randomly(for testing model)


-----
# Train
1. ``python generate_dataset.py``
1. ``python train.py``
1. ALL CLI arguments have default value, pls consult train.py.
-----

# Evaluation
*Since labels are random, so evaluation results just for test.*

INFO:tensorflow:Saving dict for global step 6000: 

ctcvr_accuracy = 0.961, 

ctcvr_accuracy_baseline = 0.961, 

ctcvr_auc = 0.50201297, 

ctcvr_average_loss = 0.8597174, 

ctcvr_precision = 0.0, 

ctcvr_recall = 0.0, 

ctr_accuracy = 0.61, 

ctr_accuracy_baseline = 0.61, 

ctr_auc = 0.5055086, 

ctr_average_loss = 0.8597174, 

ctr_precision = 0.0, 

ctr_recall = 0.0, 

global_step = 6000, 

loss = 0.8597172
