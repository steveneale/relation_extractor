#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
'utils.py' (relation_extractor/src/preprocessing)

Preprocessing utility functions

2019 Steve Neale <steveneale3000@gmail.com>

"""

import re


def clean_text(text):

    text = text.lower()
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    return text.strip()
