## Recommender

In this repo, I will post some popular models in Recommendation using deep learning.

### MODEL
1. **Deep Cross Network (DCN)**
>Ruoxi Wang, Bin Fu, Gang Fu, Mingliang Wang, Deep & Cross Network for Ad Click Predictions, 
http://arxiv.org/abs/1708.05123

>Deep & Cross Network (DCN) keeps the benefits of a DNN model, and beyond that, it introduces a novel cross network that is more efficient in learning certain bounded-degree feature interactions. 

DCN is implemented using customized ``estimator``. This high-level api in tensorflow is very neat.


### EVALUATE METRIC
1. **weighted, per-user loss**
> Paul Covington, Jay Adams, Emre Sargin Google, Deep Neural Networks for YouTube Recommendations,
http://dl.acm.org/citation.cfm?doid=2959100.2959190

>The value shown for each configuration (“weighted, per-user loss”) was obtained by considering both positive (clicked) and negative (unclicked) impressions shown to a user on a single page. We first score these two impressions with our model. If the negative impression receives a higher score than the posi- tive impression, then we consider the positive impression’s watch time to be mispredicted watch time. Weighted, per- user loss is then the total amount mispredicted watch time as a fraction of total watch time over heldout impression pairs.

* using Spark to group by each user, get their labels, predictions and weight(watch time, see paper4.2), respectively
* weighted loss: construct (pos, neg) pair and then calc metric based on pairs for each user.
