# -*- coding: utf-8 -*-

import tensorflow as tf


def get_tf_config():
    '''
        This function returns tensorflow config.
        Memory used by GPU is bounded by 0.4 * max memory
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    return config


def fc_network(input, num_layers, num_units):
    '''
        This function builds simple fully-connected
        network with given number of layers and number
        of units in each layer. Activation function
        is ReLU.
        
        Params:
            - input: Tensor -- input to network
            - num_layers: int -- number of layers in network
            - num_units: int -- number of units in each layer
            
        Returns:
            - out: Tensor -- input transformed by network
    '''
    out = input
    
    for i in range(num_layers):
        out = tf.layers.dense(out, 
                              num_units, 
                              activation=tf.nn.relu,
                              use_bias=False,
                              name='dense_{}'.format(i))
        
    return out