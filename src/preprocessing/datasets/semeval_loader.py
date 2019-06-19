#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
'semeval_loader.py' (relation_extractor/src/preprocessing/datasets)

Class for loading the SemEval dataset for relation extraction

2019 Steve Neale <steveneale3000@gmail.com>

"""

import numpy as np

import tensorflow as tf

from tflearn.data_utils import VocabularyProcessor

from resources.labels import semeval_classes
from src.io.utils import load_from_file
from src.preprocessing import Tokeniser
from src.preprocessing.utils import clean_text


class SemevalLoader():

    def __init__(self):
        
        self.x = []
        self.y = []
        self.maximum_sentence_length = 0
        self.vocab_processor = None


    def load(self, dataset_path, vocab_processor=None):

        lines = load_from_file(dataset_path, stripped=True)
        for i in range(0, len(lines), 4):
            if lines[i] != "":
                _, sentence = lines[i].split("\t")
                relation = semeval_classes[lines[i+1]] 
                self.get_x_and_y_from_sentence_and_relation(sentence, relation)
        self.process_vocabulary(vocab_processor=vocab_processor)
        self.y = tf.keras.utils.to_categorical(self.y, num_classes=len(np.unique(self.y)))
        return self


    def get_x_and_y_from_sentence_and_relation(self, sentence, relation):

        sentence = self.process_sentence_and_update_maximum_length(sentence[1:-1])
        self.x.append(sentence)
        self.y.append(relation)


    def process_sentence_and_update_maximum_length(self, sentence):

        sentence = sentence.replace("<e1>", "_e1left_")
        sentence = sentence.replace("</e1>", "_e1right_")
        sentence = sentence.replace("<e2>", "_e2left_")
        sentence = sentence.replace("</e2>", "_e2right_")
        sentence = clean_text(sentence)
        tokenised_sentence = self.tokenise_sentence_and_update_maximum_length(sentence)
        return tokenised_sentence


    def tokenise_sentence_and_update_maximum_length(self, sentence):

        tokens = Tokeniser().tokenise(sentence)
        if len(tokens) > self.maximum_sentence_length:
            self.maximum_sentence_length = len(tokens)
        return " ".join(tokens)


    def process_vocabulary(self, vocab_processor=None):

        if vocab_processor == None:
            self.vocab_processor = VocabularyProcessor(self.maximum_sentence_length).fit(self.x)
            self.x = np.array(list(self.vocab_processor.transform(self.x)))
        else:
            self.x = np.array(list(vocab_processor.transform(self.x)))


