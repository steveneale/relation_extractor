#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
'dataset_loader.py' (relation_extractor/src/preprocessing)

Class for loading relation extraction datasets

2019 Steve Neale <steveneale3000@gmail.com>

"""

import importlib


class DatasetLoader():

    def __init__(self, dataset="semeval"):

        self.dataset = dataset
        self.data = None


    def load(self, dataset_path, *args):

        loader_module = importlib.import_module("src.preprocessing.datasets")
        loader = getattr(loader_module, "{}Loader".format(self.dataset.title()))
        self.data = loader().load(dataset_path, *args)
        return self.data