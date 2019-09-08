# Recommender-In-Detail

As its name suggests,`Recommender-In-Detail` is a package which offers detailed implementations of state-of-the-art techniques and basic methods in recommendation. Most of them could be used in production environments through simple modifications. The structure of the package is as follows:

1. `models`: contains many sub-package which implements recommendation methods.
2. `metrics`: some metrics to evaluate recommendation.
3. `dataset`: dataset to used in `examples`.
4. `examples`: examples to test our implementation in `models` by using dataset in `dataset` and metrics in `metrics`.
5. `utils`: some useful tools, e.g., embedding visualization, and etc.

## Implemented Models

| MODELS| TAG | PAPERS |
|  :---  | :---  | :--:  |
| DeepWalk  | Embedding          | [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652) |
| Node2Vec | Embedding          | [node2vec: Scalable Feature Learning for Networks](<https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf>) |
| FaceBook GBDT+LR | Embedding/CTR      | [Practical lessons from predicting clicks on Ads at Facebook]() |
| Wide&Deep | Deep CTR | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792) |
| Youtube   Candidate Generation | Embedding/Deep CTR | [Deep Neural Networks for YouTube Recommendations]() |
| Youtube    Ranking | Deep CTR | [Deep Neural Networks for YouTube Recommendations]() |
| DCN | Deep CTR | [Deep & Cross Network for Ad Click Predictions](<https://arxiv.org/pdf/1708.05123.pdf>) |
| PNN | Deep CTR | [Product-based Neural Networks for User Response Prediction](https://arxiv.org/abs/1611.00144) |
| DeepFM | Deep CTR | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](<https://arxiv.org/abs/1703.04247>) |
| ESMM | Deep CTR /CVR | [Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](<https://arxiv.org/abs/1804.07931>) |
| DIN | Deep CTR | [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978) |
| XdeepFM | Deep CTR | [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/abs/1803.05170) |
| NCF | Deep MF | [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031) |
| LFM | MF/CF | [funkSVD](https://sifter.org/~simon/journal/20061211.html) |
| W_ALS | MF/CF | [CollaborativeFiltering for Implicit Feedback Datasets](https://www.researchgate.net/publication/220765111_Collaborative_Filtering_for_Implicit_Feedback_Datasets?ev=pub_cit) |
| Fast_ALS | MF/CF |  |
| BPR | MF/CF | [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618) |
| VBPR | MF/CF | [VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1510.01784) |
| LSA | MF/topic model | 单元格 |
| pLSA | topic model | 单元格 |
| LDA | topic model | 单元格 |
| Item_based | CF | [Item-based collaborative filtering recommendation algorithms]() |
| Personal Rank | CF |  |
| Item2Vec | Embedding          | [Item2Vec:Neural Item Embedding for Collaborative Filtering](https://arxiv.org/abs/1603.04259) |
| AutoEncoder |  |  |
| Tompson Sampling | Bandit |  |
| UCB | Bandit |  |
| LinUCB | Bandit |  |
| Bert | NLP |  |
| Causal Embedding | MF/Cause |  |

## Metrics

| METRICS | PAPERS |
| ---- | ------ |
| ndcg@K |        |
|      |        |
|      |        |
|      |      |




## Dataset

https://tianchi.aliyun.com/dataset/

## Utils





