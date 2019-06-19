#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
'optimiser.py' (relation_extractor/src/training)

Optimisation algorithm class

2019 Steve Neale <steveneale3000@gmail.com>

"""

import os

import tensorflow as tf


class Optimiser():

    def __init__(self, algorithm="adadelta", learning_rate=0.001):

        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.optimiser = self.get_optimiser()


    def get_optimiser(self):

        optimiser = tf.train.AdadeltaOptimizer(self.learning_rate)
        return optimiser