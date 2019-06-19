#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
'train.py' (relation_extractor/src)

Train a bi-directional LSTM model for relation extraction

2019 Steve Neale <steveneale3000@gmail.com>

"""

import tensorflow as tf

from src.models import AttentionBLSTM
from src.preprocessing import DatasetLoader
from src.training import Trainer

tf.logging.set_verbosity(tf.logging.ERROR)


def train_new_model(arguments):

    train_data = get_training_data(arguments.data)
    test_data = get_test_data(arguments.testdata, train_data.vocab_processor)
    test_data.x = tf.keras.preprocessing.sequence.pad_sequences(test_data.x, 
                                                                maxlen=train_data.maximum_sentence_length,
                                                                padding="post",
                                                                truncating="post")
    model = get_model_with_parameters(train_data.maximum_sentence_length,
                                      train_data.y.shape[1],
                                      len(train_data.vocab_processor.vocabulary_),
                                      embedding_size=(50 if arguments.embeddings == None else arguments.embeddings[1]))
    trainer = Trainer(epochs=arguments.epochs, batch_size=arguments.batchsize)
    embeddings = get_embeddings(arguments.embeddings, train_data.vocab_processor.vocabulary_) if arguments.embeddings != None else None
    trainer.train(model, train_data, test_data, arguments.name, embeddings=embeddings)


def get_training_data(data_arguments):

    train_data_path, train_data_type = data_arguments
    return DatasetLoader(dataset=train_data_type).load(train_data_path)


def get_test_data(test_data_arguments, vocab_processor):

    test_data_path, test_data_type = test_data_arguments
    return DatasetLoader(dataset=test_data_type).load(test_data_path, vocab_processor)


def get_model_with_parameters(maximum_sequence_length, number_of_classes, vocabulary_size, embedding_size=50):

    model = AttentionBLSTM(maximum_sequence_length=maximum_sequence_length,
                           number_of_classes=number_of_classes,
                           hidden_dimension=100,
                           vocabulary_size=vocabulary_size,
                           embedding_size=embedding_size)
    model.construct()
    return model


def get_embeddings(embedding_arguments, vocabulary):

    embeddings_path, dimension = embedding_arguments
    return DatasetLoader(dataset="glove").load(embeddings_path, int(dimension), vocabulary)



