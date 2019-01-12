import tensorflow as tf
from tensorflow.contrib.lookup import index_table_from_file


class BPRDynamicSampler(object):
    """Baysesian Personal Ranking (BPR) provides a pair-wise criterion for matrix factorization training.

    This class can generate a tf.data.dataset for BPR dataset with dynamic negative sampling.
    NOTE that generated dataset avoid using py_func to ensure good parallelism (num_parallel_calls) by CPU.
    Also, when negative sampling, the negative item is checked that it is an item that the user never interacted.
    Lastly, ID-index mapping is inserted as long as user_list file and item_list file is provided.

    In Alfredo L´ainez Rodrigo and Luke de Oliveira, Distributed Bayesian Personalized Ranking in Spark:

    BPR define training set as:
        D:= {(u, i , j) | i in {I_u^+} and j in  {I - I_u^+}},
        which implies D is a set of all (user, item+, item-) triples.

    Considering that we may have an enormous number of (user, items+), and that we need to sample
    negative elements for every user to effectively learning its ranking.
    (In original BPR paper of Rendle, the mentioned bootstrap sampling can not get good performance in my test)

    Params:
    user_list_file: files record all user_ID (no need start from zero), one ID is recorded in a line.
    item_list_file: files record all item_ID (no need start from zero), one ID is recorded in a line.
    user_interactive_items: dict, key is user_ID, value is list of item_IDs that the user interacted with.
                            Like: {320480545: [9943255, 9992345, ...,98435453]}
                            This is used to check whether negative item is actually negative or not,
                            you can get this dict through grouping data_file by user_ID.
    data_file: csv file, triple of rating (user, item, score) in a line.
               THIS CSV FILE CAN BE USED FOR TRAINING WEIGHTED-ALS IN SPARK (as benchmark) DIRECTLY.
               Like: 42353425,6546745,3.0
                     65474785,53455436,2.0

    Usage:
    >>> user_list_file, item_list_file="user_list.voc", "item_list.voc"
    >>> data_file = "train.csv"
    >>> user_items = {53454: [342543, 7585454, 432543, 9867645], 435632: [423432]}
    >>> sess = tf.InteractiveSession()
    >>> sampler = BPRDynamicSampler(user_items, user_list_file, item_list_file, sess)
    >>> bpr_dataset_iter = sampler.dataset(data_file, 128, True)
    >>> for epoch in range(100):
    >>>     sess.run(bpr_dataset_iter.initializer)
    >>>     sess.run(bpr_dataset_iter.get_next())
    """

    def __init__(self, user_interactive_items, user_list_file, item_list_file, sess):
        self.user_items = user_interactive_items
        self.item_table = index_table_from_file(user_list_file)
        self.user_table = index_table_from_file(item_list_file)
        self.generate_sparse_tensor_table(sess)

    def set_sparse_tensor_index_value(self, user, items):
        user_index = self.user_table.lookup(user)
        items_index = self.item_table.lookup(items)
        for i in range(items.get_shape().as_list()[0]):
            yield [user_index, tf.cast(tf.constant(i), tf.int64)], items_index[i]

    @staticmethod
    def lookup_sparse_tensor_by_index(sparse_tensor, index):
        row, col = sparse_tensor.get_shape().as_list()
        ret = tf.sparse_slice(sparse_tensor, [index, 0], [1, col]).values
        ret = tf.cast(ret, tf.int64)
        return ret

    def generate_sparse_tensor_table(self, sess):
        index = []
        value = []
        for user, items in self.user_items.items():
            u = tf.constant(str(user))
            i = tf.constant(list(map(str, items)))
            d = self.set_sparse_tensor_index_value(u, i)
            for k, v in d:
                index.append(k)
                value.append(v)
        sess.run(tf.tables_initializer())
        self.num_user = sess.run(self.user_table.size())
        self.num_item = sess.run(self.item_table.size())
        index, value = sess.run([index, value])
        self.table = tf.SparseTensor(index, value, [self.num_user, self.num_item])

    def dataset(self, data_file, batch_size, shuffle):
        def parse(line):
            _CSV_COLUMN_DEFAULTS = [[''], [''], [0.0]]
            columns = tf.decode_csv(line, record_defaults=_CSV_COLUMN_DEFAULTS)
            columns.pop()
            u = columns[0]
            u = self.user_table.lookup(u)
            interactive_items = self.lookup_sparse_tensor_by_index(self.table, u)
            item_i = columns[1]
            item_i = self.item_table.lookup(item_i)
            item_j = tf.random_uniform(shape=[], minval=0, maxval=self.num_item, dtype=tf.int64)

            def cond(item_j, interactive_items):
                return tf.reduce_any(tf.equal(interactive_items, item_j))

            def body(item_j, interactive_items):
                item_j = tf.random_uniform(shape=[], minval=0, maxval=self.num_item, dtype=tf.int64)
                return item_j, interactive_items

            j, _ = tf.while_loop(cond, body, [item_j, interactive_items])
            columns.append(j)
            features = {'u': u, 'i': item_i, 'j': j}
            return features

        dataset = tf.data.TextLineDataset(data_file)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=20000)
        dataset = dataset.map(parse, num_parallel_calls=8)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(batch_size * 8)
        iterator = dataset.make_initializable_iterator()
        return iterator
