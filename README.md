# 中文文本分类

之前做的[中文文本分类](https://github.com/sun830910/Text_Classification)仅是将中文进行分词后以one-hot形式进行文本分类，这次尝试加载静态词向量后对中文语料进行分类，并在不更动模型结构下查看分类效果的差距。

## 环境

Tensorflow

keras

Python3

## 数据集

使用THUCNews进行训练与测试（由于数据集太大，无法上传到Github，可自行下载）
百度网盘:链接: https://pan.baidu.com/s/1nD9ej_waIPpk_GITTbgXGA 密码: 3swf

数据集划分如下：
训练集cnews.train.txt 50000条
验证集cnews.val.txt 5000条
测试集cnews.test.txt 10000条
共分为10个类别："体育","财经","房产","家居","教育","科技","时尚","时政","游戏","娱乐"。 cnews.vocab.txt为词汇表，字符级，大小为5000。

## 预训练词向量
随便选了网络上一个基于维基百科训练的词向量，叫做"sgns.wiki.word"，具体谁训练的已经忘了，有需要的话issue留言我再上传至网盘。
## 文件说明

将数据集中的四份数据存放至data资料夹中

src中的文件为代码存放资料夹：

utils.py:加载数据与预处理相关函数。

model.py:模型结构主体。

main.py:主函数文件

## 结果

待补充