#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
'trainer.py' (relation_extractor/src/training)

Class for training relation extraction models

2019 Steve Neale <steveneale3000@gmail.com>

"""

import math

from typing import Tuple

import numpy as np

import tensorflow as tf
from tensorflow.data import Dataset as TFDataset
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Optimizer, Adadelta

from sklearn.metrics import f1_score

from src.preprocessing import DatasetLoader
from src.training import Optimiser
from src.io.utils import create_directory


class NewTrainer(object):

    def __init__(self, seed=None):
        self.seed = seed

    def train(self,
              train_path_and_dataset: Tuple[str, str],
              model_config=None,
              batch_size: int = None,
              model_name: str = None):
        train_path, train_dataset = train_path_and_dataset
        _ = self._load_dataset(train_path, train_dataset, batch_size=batch_size)
        loss = self._define_loss()
        optimiser = self._define_optimiser()

    def _load_dataset(self, data_path: str, dataset: str, shuffle: bool = True, batch_size: int = None) -> TFDataset:
        loaded_data = DatasetLoader(dataset=dataset).load(data_path)
        dataset = TFDataset.from_tensor_slices(loaded_data.x, loaded_data.y)
        if shuffle:
            dataset.shuffle(len(loaded_data.x), seed=self.seed)
        if batch_size:
            dataset.batch(batch_size)
        return dataset

    def _define_loss(self) -> SparseCategoricalCrossentropy:
        return SparseCategoricalCrossentropy(from_logits=True)

    def _define_optimiser(self) -> Optimizer:
        return Adadelta(learning_rate=1.0)


class Trainer():

    def __init__(self, epochs, batch_size=1):

        # Tensorflow objects (session, model and optimisation)
        self.session = None
        self.model = None
        self.model_name = None
        self.optimisation = None
        # Training data / parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.batches = None
        self.batch_count = 1
        # Test data
        self.test_data = None
        # Training values (loss / accuracy)
        self.training_loss = []
        self.training_accuracy = []
        self.test_loss = []
        self.test_accuracy = []
        # Output directories / saver
        self.output_directory = None
        self.checkpoint_directory = None
        self.saver = None
        # Metrics
        self.best_f1 = 0.0


    def train(self, model, train_data, test_data, name, embeddings=None):

        self.setup_directories_and_saver(name, train_data.vocab_processor)
        self.setup_train_and_test_data(train_data, test_data)
        self.setup_model_and_optimiser(model)
        self.initialise_tensorflow_session_and_variables(embeddings)
        self.train_model_on_batches_using_optimiser()
        self.session.close()


    def setup_directories_and_saver(self, name, vocab_processor):

        self.model_name = name
        self.output_directory = create_directory("output/{}".format(self.model_name))
        self.checkpoint_directory = create_directory("{}/checkpoints".format(self.output_directory))
        vocab_processor.save("{}/vocab".format(self.output_directory))
        self.saver = tf.train.Saver(tf.global_variables())


    def setup_train_and_test_data(self, train_data, test_data):

        self.batches = self.batch_iterator(train_data)
        self.test_data = test_data


    def setup_model_and_optimiser(self, model):

        self.model = model
        optimiser = Optimiser(algorithm="adadelta", learning_rate=1.0).optimiser
        gradients, variables = zip(*optimiser.compute_gradients(self.model.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 4.0)
        self.optimisation = optimiser.apply_gradients(zip(gradients, variables))


    def initialise_tensorflow_session_and_variables(self, embeddings=None):

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        if type(embeddings) != None:
            self.session.run(self.model.embedding_matrix.assign(embeddings))


    def batch_iterator(self, train_data, shuffle=True):

        self.batch_count = math.ceil(train_data.x.shape[0] / self.batch_size)
        data = np.array(list(zip(train_data.x, train_data.y)))
        data_size = len(data)
        batches_per_epoch = math.ceil(data_size / self.batch_size)
        for epoch in range(self.epochs):
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                data = data[shuffle_indices]
            for batch_id in range(batches_per_epoch):
                start_index = batch_id * self.batch_size
                end_index = min((batch_id+1) * self.batch_size, data_size)
                yield data[start_index:end_index]


    def train_model_on_batches_using_optimiser(self):

        epoch_id, batch_id = 1, 0
        for batch in self.batches:
            x_batch, y_batch = zip(*batch)
            loss, accuracy = self.optimise_on_batch(x_batch, y_batch)
            if batch_id % int(self.batch_count / 10) == 0 and batch_id > 0:
                if batch_id % self.batch_count == 0 and batch_id > 0:
                    self.training_loss.append(loss)
                    self.training_accuracy.append(accuracy)
                    test_loss, test_accuracy, test_f1 = self.test_model()
                    print("Epoch {} of {} | Test loss: {:.2f} | Test accuracy: {:.2f} | Test F1: {}\n".format(epoch_id, self.epochs, test_loss, test_accuracy, test_f1))     
                    epoch_id += 1
                    batch_id = 0
                else:
                    print("--- {}% of batches from epoch {} complete | Train loss: {:.2f} | Train accuracy: {:.2f}".format(int(((100 / self.batch_count) * batch_id)), epoch_id, loss, accuracy))
            batch_id += 1


    def optimise_on_batch(self, x_batch, y_batch):

        training_dict = { self.model.x_train: x_batch, 
                          self.model.y_train: y_batch,
                          self.model.embedding_dropout_prob: 0.7,
                          self.model.cell_dropout_prob: 0.7,
                          self.model.dropout_prob: 0.5 }
        _, loss, accuracy = self.session.run([self.optimisation, self.model.loss, self.model.accuracy], feed_dict=training_dict)
        return loss, accuracy


    def test_model(self):

        test_dict = { self.model.x_train: self.test_data.x,
                      self.model.y_train: self.test_data.y,
                      self.model.embedding_dropout_prob: 1.0,
                      self.model.cell_dropout_prob: 1.0,
                      self.model.dropout_prob: 1.0 }

        loss, accuracy, predictions = self.session.run([self.model.loss, self.model.accuracy, self.model.predictions], feed_dict=test_dict)
        self.test_loss.append(loss)
        self.test_accuracy.append(accuracy)
        f1 = f1_score(np.argmax(self.test_data.y, axis=1), predictions, labels=np.array(range(1, 19)), average="macro")
        if self.best_f1 < f1:
            self.best_f1 = f1
            saved_checkpoint_path = self.saver.save(self.session, self.checkpoint_directory + "/model-{:.3g}-f1".format(self.best_f1))
            print("Saved model checkpoint to {}".format(saved_checkpoint_path))
        return loss, accuracy, f1
