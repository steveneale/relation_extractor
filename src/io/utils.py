#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'utils.py' (relation_extractor/src/io)

Input/output utility functions

2019 Steve Neale <steveneale3000@gmail.com>

"""

import os


def load_from_file(file_path, lines=True, stripped=False):
    with open(file_path, "r", encoding="utf-8") as loaded_file:
        if lines is True:
            return [line.strip() for line in loaded_file.read().splitlines()] \
                    if stripped is True else loaded_file.read().splitlines()
        else:
            return loaded_file.read()


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path
