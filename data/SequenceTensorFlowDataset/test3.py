import tensorflow as tf

feature_parse = {}


def _float(p):
    if not p:
        return 0.0
    else:
        return float(p)


def _week(p):
    p = p.encode('utf-8')
    return _byte_feature(value=[p])


def _week_list(p):
    p = _float(p)
    return _float_feature(value=[p])


def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write():
    file = "./records2"
    writer = tf.python_io.TFRecordWriter(file)

    context_features = {}
    sequence_features = {}

    with tf.gfile.FastGFile('abcd', 'r') as f:
        for line in f:
            line = line.strip('\n')
            lines = line.split('|')

            label = lines[0]
            week = lines[1]
            week_weight = lines[2]

            # features
            # label
            label = int(label)
            # week_list
            week_list = week.split(',')
            week_list = list(map(_week, week_list))
            # week_weight
            week_weight_list = week_weight.split(',')
            week_weight_list = list(map(_week_list, week_weight_list))

            example = tf.train.SequenceExample(
                context=tf.train.Features(feature={
                    'label': _int_feature([label])}),
                feature_lists=tf.train.FeatureLists(feature_list={
                    'week_list': tf.train.FeatureList(feature=week_list),
                    'week_weight': tf.train.FeatureList(feature=week_weight_list)}
                ))
            print()
            writer.write(example.SerializeToString())
    writer.close()


def parse(serialized_example):
    context_feature = {
        'label': tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "week_list": tf.FixedLenSequenceFeature([], dtype=tf.string),
        'week_weight': tf.FixedLenSequenceFeature([], dtype=tf.float32)
    }

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_feature,
        sequence_features=sequence_features
    )

    labels = context_parsed['label']
    week_list = sequence_parsed['week_list']
    week_weight = sequence_parsed['week_weight']
    return labels, week_list, week_weight


def batched_data(single_example_parser, batch_size, padded_shapes):
    dataset = tf.data.TFRecordDataset('records2') \
        .map(single_example_parser) \
        .padded_batch(batch_size, padded_shapes=padded_shapes)

    iterator = dataset.make_one_shot_iterator()
    labels, week_list, week_weight = iterator.get_next()
    return labels, week_list, week_weight


def input_fn(data_file, batch_size):
    """Generate an input function for the Estimator."""
    assert tf.gfile.Exists(data_file), (
            '%s not found. Please make sure you have set both arguments --train_data and --test_data.' % data_file)

    def _return_cols(context_parsed, sequence_parsed):
        global feature_parse
        feature_parse = dict(list(context_parsed.items()) + list(sequence_parsed.items()))
        return list(feature_parse.values())

    def parse(serialized_example):
        context_feature = {'label': tf.FixedLenFeature([1], dtype=tf.int64)}
        sequence_features = {
            "week_list": tf.FixedLenSequenceFeature([], dtype=tf.string),
            'week_weight': tf.FixedLenSequenceFeature([], dtype=tf.float32)}
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=serialized_example,
            context_features=context_feature,
            sequence_features=sequence_features)
        cols = _return_cols(context_parsed, sequence_parsed)
        return cols

    def watch_features(*features):
        cols = list(feature_parse.keys())
        features = dict(zip(cols, features))
        watch_label = features.pop('label')
        # reward_label = features.pop('reward_label')
        return features, watch_label

    dataset = tf.data.TFRecordDataset(data_file) \
        .map(parse) \
        .padded_batch(batch_size, ([7], [1], [7])) \
        .map(watch_features)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


with tf.Session() as sess:
    features, labels = input_fn('./records2', 4)
    print(sess.run([features, labels]))
