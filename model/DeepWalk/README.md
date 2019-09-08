# DeepWalk

This is a refactor of [phanein](https://github.com/phanein)/**deepwalk**, which is implemented using pure `pySpark`. Since

1. Data are stored in HDFS and could be loaded by Spark.
2. Distributed Computing of Spark helps generating random walks in parallel.
3. Spark.mllib support pre-canned Skip-gram.

[中文博客:DeepWalk](https://yinyajun.github.io/ML-Recommend/deep_walk/).

 **Chinese blog**: [DeepWalk](https://yinyajun.github.io/ML-Recommend/deep_walk/).

## Notes

Details of DeepWalk, please see paper[1],[2].

Note that paper[1] construct **unweighted graph** and adopt **standard random walk** where the transition probability of each node is uniform. 

According to paper[2], user behaviors could construct **weighted graph**. Edge weight means occurrence of adjacent items in user behaviors log. In addition, **biased random walk** is adopted in this paper. The transition probability is depend on outgoing edges' weight instead of  equal probability.

**This package follows paper[2]'s methods and it could deteriorates to standard Random Walk in  paper[1] easily.**

## Usage

```python
from model.DeepWalk import DeepWalk

def get_input_rdd():
    # ...
    return rdd
    
dataset = get_input_rdd()
m = DeepWalk(num_paths=50)
vectors = m.fit(dataset)
m.save_vectors(path="/tmp/random_walk.emb", vectors=vectors)
```

## Params

### Model Params

1. alpha: Restart Probability (default: 0.0)
2. learning_rate: Skip gram learning rate (default: 0.025)
3. max_active_num: Avoid Spam User, max impressed items num per user (default: 200)
4. min_count: Min count of a vertex in corpus (default: 50)
5. num_iter: Skip gram iteration nums (default: 1)
6. num_partitions: Num partitions of training skip gram (default: 10)
7. num_paths: The nums of paths for a vertex (default: 20)
8. num_workers: Parallelism for generation walks (default: 400)
9. path_length: The maximum length of a path (default: 50)
10. pre_process: Whether use processing function or not (default: True)
11. session_duration: The duration for session segment (default: 3600.0)
12. vector_size: Low latent vector size (default: 50)
13. window_size: Window size for target vertex (default: 5)

### Input

1. If `pre_process` is `True`, `DeepWalk.preprocessing` helps you pre-processing.
    All you need is to supply RDD with a format likes: [user,item,timestamp],

2. If `pre_process` is `False`, you should supply RDD of adjacency table.
    Format like:[item1,(item2,item3,...),(weight1, weitht2, weight3,...)]
   (How to get ? you could see `DeepWalk.preprocessing` )
3. **If weights in adjacency table are set as `None`, it deteriorates to standard Random Walk.**

## Example
see `example/deepwalk.py`

## Papers

1. [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652)
2. [Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba](https://arxiv.org/abs/1803.02349)
