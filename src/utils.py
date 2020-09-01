# -*- coding: utf-8 -*-

"""
Created on 2020-08-25 14:28
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""


import re
import os
import jieba

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


def get_training_embedding():
    file_path = '../data/cnews.train.txt'
    if not os.path.exists(file_path):
        print('file is not exist')
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        spliteds = [list(jieba.cut(line)) for line in lines]

        print(spliteds)
        # labels = [splited[0] for splited in spliteds]
        # texts = [splited[1] for splited in spliteds]

    # print(labels)

get_training_embedding()