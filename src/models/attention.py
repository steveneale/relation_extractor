#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
'attention.py'

Attention function

2019 Steve Neale <steveneale3000@gmail.com>

"""

import tensorflow as tf


def attention(input_values):

    parameter_vector = tf.get_variable("parameter_vector", 
                                       [input_values.shape[2].value],
                                       initializer=tf.keras.initializers.glorot_normal())
    values = tf.tanh(input_values)
    alphas = tf.nn.softmax(tf.tensordot(values, parameter_vector, axes=1, name="value_param_dot"), name="alphas")
    output = tf.reduce_sum(input_values * tf.expand_dims(alphas, -1), 1)
    output = tf.tanh(output)

    return output, alphas

