仿照wide_deep官方模型写法的deepFM模型，使用feature_column作特征处理，使用estimator等高阶api.
* 不用将数据集转换为libsvm格式.
* 支持multi-hot类型特征，例如用户的历史消费物品序列.
* 暂不支持数值型特征的交叉.
【特征说明】
参考train.py.
1. 原始特征分为两种：
* 数值特征(dense)：没啥好说的
* 离散特征(sparse)：id类特征（vocabulary size特别大，使用特征hash），非id类特征（vocabulary size在几个到几百个之间，直接one-hot编码）
特别的，当多个不定长离散特征组成一个序列，直接编码就是multi-hot特征，非常常见。
2. 简要叙述官方wide_deep
原生的wide部分由于其实现方式，直接支持sparse tensor的输入；
而dnn部分只支持dense tensor输入。
这里直接copy官方写法，也分为两部分输入。
【DeepFM简要说明】
* 按理说，deepFM的一阶项和二阶项应该都是相同的输入，分为两部分是仿照wide_deep写法，能够直接复用linear model代码。
* 数值特征目前还不支持。
* 将离散特征直接传给linear_feature_columns，作为一阶项的输入。
* 将离散特征通过embedding变为dense tensor后传给dnn_feature_columns，作为二阶项和dnn的输入。
* 实质上，一阶项和二阶项的输入是相同的。
* 为了稍微利用数值特征，将数值特征分桶后用在了linear部分，而二阶项和dnn部分目前没有使用数值特征（todo）