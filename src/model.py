# -*- coding: utf-8 -*-

"""
Created on 2020-09-01 16:03
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""

import keras
from keras import layers


class LSTMclf(object):
    def __init__(self):
        self.num_class = 10
        self.seq_length = 600
        self.embedding_dim = 300
        self.vocab_size = 352217

    def model(self):
        model_input = layers.Input((self.seq_length ,self.embedding_dim))

        LSTMed = layers.LSTM(256)(model_input)
        full_connect = layers.Dense(128)(LSTMed)
        droped = layers.Dropout(0.5)(full_connect)
        model_output = layers.Dense(self.num_class, activation="softmax")(droped)
        model = keras.models.Model(inputs=model_input, outputs=model_output)
        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["acc"])
        model.summary()
        return model


if __name__ == '__main__':
    test = LSTMclf()
    test.model()
