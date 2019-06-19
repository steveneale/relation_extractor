#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
'labels.py' (relation_extractor/resources)

Class label mappings for relation extraction datasets

2019 Steve Neale <steveneale3000@gmail.com>

"""

semeval_classes = { "Other": 0,
                    "Message-Topic(e1,e2)": 1,
                    "Message-Topic(e2,e1)": 2,
                    "Product-Producer(e1,e2)": 3,
                    "Product-Producer(e2,e1)": 4,
                    "Instrument-Agency(e1,e2)": 5,
                    "Instrument-Agency(e2,e1)": 6,
                    "Entity-Destination(e1,e2)": 7,
                    "Entity-Destination(e2,e1)": 8,
                    "Cause-Effect(e1,e2)": 9,
                    "Cause-Effect(e2,e1)": 10,
                    "Component-Whole(e1,e2)": 11,
                    "Component-Whole(e2,e1)": 12,
                    "Entity-Origin(e1,e2)": 13,
                    "Entity-Origin(e2,e1)": 14,
                    "Member-Collection(e1,e2)": 15,
                    "Member-Collection(e2,e1)": 16,
                    "Content-Container(e1,e2)": 17,
                    "Content-Container(e2,e1)": 18 }