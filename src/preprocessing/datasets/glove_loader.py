#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
'glove_loader.py' (relation_extractor/src/preprocessing/datasets)

Class for loading pre-trained GloVe word embeddings

2019 Steve Neale <steveneale3000@gmail.com>

"""

import numpy as np

from src.io.utils import load_from_file


class GloveLoader():

    def __init__(self):
        
        self.matrix = None


    def load(self, embeddings_path, dimension, vocabulary):

        self.matrix = np.random.randn(len(vocabulary), dimension).astype(np.float32) / np.sqrt(len(vocabulary))
        lines = load_from_file(embeddings_path, stripped=True)
        for line in lines:
            split = line.split(" ")
            word = split[0]
            vector = np.asarray(split[1:], dtype="float32")
            _id = vocabulary.get(word)
            if _id != 0:
                self.matrix[_id] = vector
        return self.matrix