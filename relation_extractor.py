#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'relation_extractor.py'

2019 Steve Neale <steveneale3000@gmail.com>
"""

import sys
import argparse

import src.train as training


def train(arguments):
    training.train_new_model(arguments)


def parse_training_arguments(args):
    parser = argparse.ArgumentParser(description="relation_extractor.py (train) - \
                                                  Train a bi-directional LSTM model for relation extraction")
    optional = parser._action_groups.pop()
    required = parser.add_argument_group("required arguments")
    required.add_argument("train")
    required.add_argument("-d", "--data",
                          help="Path to the training data, and its type (only 'semeval' currently supported)",
                          nargs=2, required=True)
    required.add_argument("-n", "--name",
                          help="Name for the relation extraction model to be trained",
                          required=True)
    required.add_argument("-e", "--epochs",
                          help="How many epochs to train for",
                          type=int, required=True)
    required.add_argument("--testdata",
                          help="Path to the test data, and its type (only 'semeval' currently supported)",
                          nargs=2, required=True)
    optional.add_argument("--embeddings",
                          help="Pre-trained word embeddings (GloVe) to use during training, and their dimensions",
                          nargs=2, required=True)
    optional.add_argument("-b", "--batchsize",
                          help="Batch size to use during training",
                          type=int, default=1, required=False)
    parser._action_groups.append(optional)
    return(parser.parse_args())


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[0] == "train":
        arguments = parse_training_arguments(args)
        train(arguments)
    else:
        print("ERROR: No arguments given. For help, please see the 'lang_detector' README")
