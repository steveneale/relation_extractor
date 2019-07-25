#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'attention_blstm.py'

Bi-directional LSTM class with attention layer

2019 Steve Neale <steveneale3000@gmail.com>

"""

import tensorflow as tf

from src.models import attention


class AttentionBLSTM():

    def __init__(self, maximum_sequence_length, number_of_classes, hidden_dimension, vocabulary_size, embedding_size):
        self.maximum_sequence_length = maximum_sequence_length
        self.number_of_classes = number_of_classes
        self.hidden_dimension = hidden_dimension
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

    def construct(self):
        self.define_inputs()
        self.define_dropout()
        self.define_initialiser()
        self.define_embedding_layer()
        self.build_bidirectional_lstm()
        self.add_attention_and_dropout()
        self.build_output_layer()
        self.define_loss()
        self.define_accuracy()

    def define_inputs(self):
        self.x_train = tf.placeholder(tf.int32, [None, self.maximum_sequence_length], name="input_samples")
        self.y_train = tf.placeholder(tf.float32, [None, self.number_of_classes], name="input_labels")

    def define_dropout(self):
        self.embedding_dropout_prob = tf.placeholder(tf.float32, name="embedding_dropout_prob")
        self.cell_dropout_prob = tf.placeholder(tf.float32, name="cell_dropout_prob")
        self.dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")

    def define_initialiser(self):
        self.initialiser = tf.keras.initializers.glorot_normal

    def define_embedding_layer(self):
        with tf.variable_scope("embedding_layer"):
            self.embedding_matrix = tf.get_variable("embedding_matrix",
                                                    shape=[self.vocabulary_size, self.embedding_size],
                                                    trainable=False)
            self.dropout_embeddings = tf.nn.dropout(self.embedding_matrix, self.embedding_dropout_prob)
            self.embedding_output = tf.nn.embedding_lookup(self.dropout_embeddings,
                                                           self.x_train,
                                                           name="embedding_output")

    def build_bidirectional_lstm(self):
        with tf.variable_scope("bidirectional_lstm_layer"):
            self.forward_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dimension, initializer=self.initialiser())
            self.fw_cell_with_dropout = tf.nn.rnn_cell.DropoutWrapper(self.forward_cell, self.cell_dropout_prob)
            self.backward_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dimension, initializer=self.initialiser())
            self.bw_cell_with_dropout = tf.nn.rnn_cell.DropoutWrapper(self.backward_cell, self.cell_dropout_prob)
            (self.fw_output, self.bw_output), self.states = \
                tf.nn.bidirectional_dynamic_rnn(self.fw_cell_with_dropout,
                                                self.bw_cell_with_dropout,
                                                self.embedding_output,
                                                dtype=tf.float32)
            self.blstm_output = tf.add(self.fw_output, self.bw_output, name="blstm_output")

    def add_attention_and_dropout(self):
        with tf.variable_scope("attention_layer"):
            self.attention, self.alphas = attention(self.blstm_output)

        with tf.variable_scope("dropout_layer"):
            self.dropout_output = tf.nn.dropout(self.attention, self.dropout_prob)

    def build_output_layer(self):
        with tf.variable_scope("output_layer"):
            self.logits = tf.layers.dense(self.dropout_output,
                                          self.number_of_classes,
                                          kernel_initializer=self.initialiser(),
                                          name="logits")
            self.predictions = tf.argmax(self.logits, axis=1, name="predictions")

    def define_loss(self):
        with tf.variable_scope("l2_loss"):
            trainable_variables = tf.trainable_variables()
            self.l2_loss = tf.add_n([tf.nn.l2_loss(variable) for variable in trainable_variables
                                     if "bias" not in variable.name]) * 0.001

        with tf.variable_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y_train)
            self.loss = tf.reduce_mean(self.losses + self.l2_loss)

    def define_accuracy(self):
        with tf.variable_scope("accuracy"):
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.y_train, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32), name="accuracy")
