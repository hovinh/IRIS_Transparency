# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 09:38:05 2017

@author: workshop
"""

import tensorflow as tf
import numpy as np
import math
import os
from iris_dataset import IRISDataset
from model import NeuralNetwork
from qii import QII


def set_up_parameter():
    global SEED, PRINT_STEPS, EPOCH, SHOULD_REUSE_MODEL, IS_EXHAUSTIVE
    # Make results reproducible
    SEED = 1234 # Make stochastic deterministic to reproduce same result
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
    
    PRINT_STEPS = 50 # Number of epoch to print out performance when training
    EPOCH = 500 # Number of epoch to train model
    SHOULD_REUSE_MODEL = False
    IS_EXHAUSTIVE = True
    
def calculate_shapley(model, X_train):
    hl_1, hl_2, hl_3, prediction = model.predict(X_train)
    numb_layers = 4
    layer_names = ['input', 'h1', 'h2', 'h3']
    layer_nodes = [4, 8, 6, 4]
    layer_data = [X_train, hl_1, hl_2, hl_3]
    predict_list = {'input': model.predict_input,
                    'h1': model.predict_hl_1,
                    'h2': model.predict_hl_2,
                    'h3': model.predict_hl_3}
    
                    
    for idx in range(numb_layers):
        numb_features = layer_nodes[idx]
        pool_samples = len(layer_data[idx])
        numb_samplings = math.factorial(numb_features)
        layer_samples = layer_data[idx]
        
        feature_list = [str(i) for i in range(1, numb_features+1)]
        influencer = QII(pool_samples, numb_samplings, layer_samples, feature_list)
        
    
        with tf.Session(graph = model.graph) as sess:
            new_saver = tf.train.import_meta_graph(os.path.join(os.getcwd(),
                                                                    'saved_model_01.meta'))
            new_saver.restore(sess, tf.train.latest_checkpoint('./'))
            layer_name = layer_names[idx]
            predict_model = predict_list[layer_name]
            X_individual = layer_samples[0]
            shapley_val = influencer.shapley(X_individual, predict_model, sess,
                                             is_exhaustive = IS_EXHAUSTIVE)
            print (shapley_val)
    
if __name__ == '__main__':
    # SET UP PARAMETERS
    set_up_parameter()
    
    # LOAD DATASET
    dataset = IRISDataset()
    X_test, X_train, y_test, y_train = dataset.split_to_train_test(seed=10)
    
    # LOAD/TRAIN MODEL
    model = NeuralNetwork()
    if (SHOULD_REUSE_MODEL == False):
        model.train(X_train, y_train, PRINT_STEPS, EPOCH)
        model.test(X_test, y_test)
    else:
        with tf.Session(graph = model.graph) as sess:
            # Initialize variables
            sess.run(tf.global_variables_initializer())
        
    # EXPERIMENTING
    calculate_shapley(model, X_train)
            
