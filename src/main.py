# -*- coding: utf-8 -*-

"""
Created on 2020-09-01 16:57
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""

from model import LSTMclf
from utils import read_data, get_embedding
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np


if __name__ == '__main__':
    embedding_dict = get_embedding()

    print('Preprocessing data...')
    training_path = '../data/cnews.train.txt'
    valid_path = '../data/cnews.val.txt'

    train_labels, train_texts = read_data(training_path)
    train_x = pad_sequences(train_texts, maxlen=600)
    train_y = to_categorical(np.asarray(train_labels))

    valid_labels, valid_texts = read_data(valid_path)
    valid_x = pad_sequences(valid_texts, maxlen=600)
    valid_y = to_categorical(np.asarray(valid_labels))

    print("Building model...")
    lstm_clf = LSTMclf()
    clf_model = lstm_clf.model()
    clf_model.fit(train_x, train_y, batch_size=32, validation_data=(valid_x, valid_y), epochs=1)
    clf_model.save('./model.h5', overwrite=True)


