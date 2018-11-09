## Recommender

In this repo, I will try to implement some basic and popular recommender. Most of models are implement with **customed estimator** which is high-level tensorflow API. It is easy for distributed training and can export saved model for tensorflow serving to predict online.

## MODEL
### Deep Cross Network (DCN)
  
>Ruoxi Wang, Bin Fu, Gang Fu, Mingliang Wang, Deep & Cross Network for Ad Click Predictions, 
https://arxiv.org/pdf/1708.05123.pdf

>Deep & Cross Network (DCN) keeps the benefits of a DNN model, and beyond that, it introduces a novel cross network that is more efficient in learning certain bounded-degree feature interactions.


### Entire Space Multi-Task Model (ESMM)

>Xiao Ma, Liqin Zhao, Guan Huang, Zhi Wang, Zelin Hu, Xiaoqiang Zhu, Kun Gai, Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate,
https://arxiv.org/pdf/1804.07931.pdf

>Entire Space Multi-task Model (ESMM) can i) modeling CVR directly over the entire space, ii) employing a feature representation transfer learning strategy.


## EVALUATE METRIC
### weighted, per-user loss
> Paul Covington, Jay Adams, Emre Sargin Google, Deep Neural Networks for YouTube Recommendations,
http://dl.acm.org/citation.cfm?doid=2959100.2959190

