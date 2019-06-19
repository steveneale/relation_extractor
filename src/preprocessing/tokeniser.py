#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
'tokeniser.py' (relation_extractor/src/preprocessing)

Tokenisation class

2019 Steve Neale <steveneale3000@gmail.com>

"""

import os
import nltk


class Tokeniser():

    def __init__(self, tokeniser="nltk_punkt"):

        self.tokeniser = tokeniser


    def tokenise(self, text):

        if not os.path.exists("resources/nltk_data/tokenizers/punkt"):
            nltk.download("punkt", download_dir="resources/nltk_data")
        return nltk.tokenize.word_tokenize(text)