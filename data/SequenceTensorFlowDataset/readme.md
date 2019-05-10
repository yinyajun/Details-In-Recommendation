这是18年9月的尝试，当时TensorFlow版本1.7。
对于序列的特征，这是一个变长的列表。
如果不固定长度的话，那么无法训练（除非batch=1）
如果对序列做截断，总觉得少了信息。
TF中有没有类似Dynamic RNN的方式，不用截断序列特征，自动将特征填充为该batch中最长的那个。
这样既有效率，又能保证信息不丢失。


探索时候发现了tf.parse_single_sequence_example,不同于常见的tf.parse_example,
这个函数可以将变长的example的PB解析。

我在示例中，展示了如何写变长特征的TFRecorder及解析，并且放入了wide&deep模型中测试。
记得模型的训练步数不高，可能是这样的写法导致IO效率不高吧，所以后来还是使用了截断序列这种方式。

代码仅供参考，找了半天找出来，当时试了好几个版本，这几个代码好像能搭配使用。

test3.py - 写特征，解析特征
test4.py - 模型
abcd - 原始数据集
recorder - tfrecorder形式数据集
