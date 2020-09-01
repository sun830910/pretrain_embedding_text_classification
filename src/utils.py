# -*- coding: utf-8 -*-

"""
Created on 2020-08-25 14:28
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""


import re
import os
import jieba
from tensorflow import keras

def to_float_array(items):
    num_arr = []
    for item in items:
        if item != '':
            num_arr.append(float(item))
    return num_arr


def get_word_embedding():
    embedding_path = '../pretrain_embedding/sgns.wiki.word'
    embedding_dict = dict()
    with open(embedding_path, 'r', encoding='utf-8') as infile:
        first_line_flag = True
        for line in infile:
            items = re.split(' ', line.strip())
            if first_line_flag:
                embedding_dim = int(items[1])
                first_line_flag = False
                continue
            if len(items) == embedding_dim + 1:
                word = items[0]
                embedding_dict.setdefault(word, to_float_array(items[1:]))
    return embedding_dict


def get_file_embedding(file_path, pretrain_embedding, max_length, num_class):
    if not os.path.exists(file_path):
        print('file is not exist')
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

    tmps = [line.strip().split('\t') for line in lines]
    labels = []
    texts = []

    categories = ["体育", "财经", "房产", "家居", "教育", "科技", "时尚", "时政", "游戏", "娱乐"]
    for tmp in tmps:
        assert len(tmp) == 2
        if tmp[0] in categories:
            labels.append(categories.index(tmp[0]))
            texts.append(list(jieba.cut(tmp[1])))

    assert len(labels) == len(texts)
    embedding = []
    for text in texts:
        text_embedding = []
        print("正在处理: {} ".format(str(text)))
        for word in text:
            if word in pretrain_embedding:
                text_embedding.append(pretrain_embedding.get(word))  # 直接忽略没出现在预训练词向量中的词
        embedding.append(text_embedding)
    assert len(labels) == len(embedding)

    x_pad = keras.preprocessing.sequence.pad_sequences(embedding, maxlen=max_length)
    y = keras.utils.to_categorical(labels, num_classes=num_class)
    return x_pad, y



if __name__ == '__main__':
    texts, labels = get_training_embedding(get_word_embedding(), max_length=600, num_class=10)
    for idx in range(len(texts)):
        print(len(texts[idx]))
        print(labels[idx])
