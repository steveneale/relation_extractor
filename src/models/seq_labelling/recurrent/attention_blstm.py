#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'attention_blstm.py'

Bi-directional LSTM class with attention layer

2019 Steve Neale <steveneale3000@gmail.com>
"""

from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dropout, Bidirectional, LSTM, Dense

from src.models.seq_labelling.recurrent import attention


class AttentionBLSTM(Model):

    def __init__(self,
                 number_of_classes: int,
                 lstm_dimension: int,
                 lstm_dropout_prob: float,
                 vocabulary_size: int,
                 embedding_size: int,
                 embedding_dropout_prob: float):
        super(AttentionBLSTM, self).__init__()
        self.number_of_classes = number_of_classes
        self.lstm_dimension = lstm_dimension
        self.lstm_dropout_prob = lstm_dropout_prob
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.embedding_dropout_prob = embedding_dropout_prob
        self._construct_model()

    def _construct_model(self):
        self.word_embedding = Embedding(self.vocabulary_size,
                                        self.embedding_size,
                                        trainable=False, name="word_embedding")
        self.embedding_dropout = Dropout(self.embedding_dropout_prob,
                                         name="embedding_dropout")
        self.bidirectional_lstm = Bidirectional(LSTM(self.lstm_dimension,
                                                     dropout=self.lstm_dropout_prob,
                                                     name="bidirectional_lstm"))
        self.final_dropout = Dropout(self.final_dropout_prob,
                                     name="final_dropout")
        self.output_logits = Dense(self.number_of_classes,
                                   name="output_logits")

    def call(self, x):
        x = self.word_embedding(x)
        x = self.embedding_dropout(x)
        x = self.bidirectional_lstm(x)
        x = attention(x)
        x = self.final_dropout(x)
        x = self.output_logits(x)
        return x
