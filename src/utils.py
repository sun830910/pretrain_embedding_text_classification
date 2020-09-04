# -*- coding: utf-8 -*-

"""
Created on 2020-09-03 16:28
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""


import os
import jieba
import numpy as np


def cate2idx(label):
    categories = ["体育", "财经", "房产", "家居", "教育", "科技", "时尚", "时政", "游戏", "娱乐"]
    if label in categories:
        return categories.index(label)
    else:
        print("Label is not in categories.")
        return


def read_data(file_path):
    if not os.path.exists(file_path):
        print("File is not exist!")
        return
    with open(file_path, encoding="utf-8") as infile:
        lines = infile.readlines()
    labels = []
    texts = []
    embedding_dict = get_embedding()
    for line in lines:
        tmp = line.strip().split('\t')
        assert len(tmp) == 2
        labels.append(cate2idx(tmp[0]))
        texts.append(text2embedding(list(jieba.cut(tmp[1])), embedding_dict))
    return labels, texts


def get_embedding_info():
    embedding_path = '../pretrain_embedding/sgns.wiki.word'
    if not os.path.exists(embedding_path):
        print("Embedding is not exits!")
        return
    with open(embedding_path, encoding='utf-8') as infile:
        lines = infile.readlines()
    firstLine_flag = True
    for line in lines:
        if firstLine_flag:
            firstLine_flag = False
            tmp = line.strip().split(' ')
            assert len(tmp) == 2
            vocab_size = int(tmp[0])
            embedding_dim = int(tmp[1])
            return vocab_size, embedding_dim


def get_embedding():
    embedding_path = '../pretrain_embedding/sgns.wiki.word'
    if not os.path.exists(embedding_path):
        print("Embedding is not exits!")
        return
    with open(embedding_path, encoding='utf-8') as infile:
        lines = infile.readlines()
    firstLine_flag = True
    embedding_dict = dict()
    vocab_size, embedding_dim = get_embedding_info()
    for line in lines:
        if firstLine_flag:
            firstLine_flag = False
            continue
        else:
            tmp = line.strip().split(' ')
            assert len(tmp) == int(embedding_dim) + 1
            embedding_dict.setdefault(tmp[0], np.asarray(tmp[1:], dtype='float64'))
    return embedding_dict


def text2embedding(words, embedding_dict):
    # print(words)
    result = []
    for word in words:
        if word in embedding_dict:
            result.append(embedding_dict.get(word))
    return result


if __name__ == '__main__':
    embedding_dict = get_embedding()  # 读取预训练词向量
    print("Preprocess")
    train_path = '../data/cnews.train.mini.txt'
    train_labels, train_texts = read_data(train_path)

