仿照wide_deep官方模型写法的deepFM模型，使用feature_column作特征处理，使用estimator等高阶api.
* 不用将数据集转换为libsvm格式.
* 支持multi-hot类型特征，例如用户的历史消费物品序列（依靠embedding_feature_column完成）
* 不支持数值型特征的交叉.

【特征说明】
参考train.py.
1. 原始特征分为两种：
* 数值特征(dense)：没啥好说的
* 离散特征(sparse)：id类特征（vocabulary size特别大，使用特征hash），非id类特征（vocabulary size在几个到几百个之间，直接one-hot编码）
特别的，当多个不定长离散特征组成一个序列，直接编码就是multi-hot特征，非常常见。

2. 关于官方wide_deep
原生的wide部分由于其实现方式，直接支持sparse tensor的输入；
而dnn部分只支持dense tensor输入。
这里直接copy官方写法，也分为两部分输入。


【简要说明】
* 按理说，deepFM的一阶项和二阶项应该都是相同的输入，分为两部分是仿照wide_deep写法，能够直接复用linear model代码。
* 论文中deepFM的输入都是sparse的离散特征，这些离散特征经过embedding输入到网络
* deep部分和wide部分都可以额外加入一些非embedding的特征（indicator，numerical）

【踩坑】
1. input_layer会将dense column转换成tensor，如果model中多次调用input_layer需要特别注意：像embedding_feature_column这种column在
input_layer中转换时会定义embedding matrix的variable，每次调用都是定义一个新的embedding matrix的variable。
2. 也就是说，多次调用input_layer，那么会有多个embedding需要更新，并不是在一个embedding matrix上更新。
3. 这里的实现是简单改写input_layer，在model中只使用一次，比较麻烦。
4. 还有个更简单的方法，使用shared_embedding来代替embedding。（查看test02.py探索其关系）
