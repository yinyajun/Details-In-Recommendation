how to use deepwalk?

```python
from pyspark.mllib.feature import Word2Vec
from deep_walk.walks import generate_walks
from deep_walk.graph import load_adjacency_list_from_spark
from deep_walk.preprocess import preprocessing

SESSION_DURATION = 3600
MIN_DURATION = 10
MAX_ACTIVE_NUM = 200


def create_graph(rdd):
    rdd = preprocessing(rdd, MIN_DURATION, MAX_ACTIVE_NUM, SESSION_DURATION)
    G = load_adjacency_list_from_spark(rdd)
    return G


def skip_gram(walks_rdd, path, vector_size, min_count, seed):
    model = Word2Vec().setVectorSize(vector_size) \
        .setMinCount(min_count) \
        .setSeed(seed) \
        .setNumPartitions(10) \
        .fit(walks_rdd)
    vectors = model.getVectors()
    return vectors


def deep_walk():
    rdd = get_source_rdd()  # format<user, item, duration, timestamp>
    G = create_graph(rdd)
    walks_rdd = generate_walks(sc, G, args.num_walks, args.walk_length, args.workers)
    skip_gram(walks_rdd, args.output, args.vec_dim, args.num_walks * 2, args.seed)

if __name__ == '__main__':
    conf = SparkConf().setAppName("deep_walk")
    sc = SparkContext(conf=conf)
    hc = HiveContext(sc)

    vec = deep_walk()

    # save vec
```