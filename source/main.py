# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 09:38:05 2017

@author: workshop
"""

import tensorflow as tf
import numpy as np
import math
import time
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
    EPOCH = 1000 # Number of epoch to train model
    SHOULD_REUSE_MODEL = True
    IS_EXHAUSTIVE = False
    
def calculate_shapley(model, X_train, sample_idx):
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
        layer_samples = layer_data[idx]
        feature_list = [str(i) for i in range(1, numb_features+1)]
        influencer = QII(layer_samples, feature_list)
        X_individual = layer_samples[sample_idx]
    
        with tf.Session(graph = model.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, model.CHECK_POINT)
            layer_name = layer_names[idx]
            predict_model = predict_list[layer_name]
            start_time = time.time()        
            print ('Calculating Shapley...')
            shapley_val = influencer.shapley(X_individual, predict_model, sess,
                                             is_exhaustive = IS_EXHAUSTIVE)
            print (shapley_val)
            print("--- %s seconds ---" % (time.time() - start_time))
            
            influence_scores = np.array([shapley_val[i] for i in feature_list])
            influence_scores = np.expand_dims(influence_scores, axis = 0)
            influencer.plot_influence_score(influence_scores, feature_list,
                                            layer_name)
            # if observed layer is not input, then we must map back
            if (idx != 0):
                weight_name_list = ['w1', 'w2', 'w3']
                weight_name_list = weight_name_list[:idx]
                influence_scores = influencer.mapback(model, X_train, influence_scores, weight_name_list)
                feature_list = [str(i) for i in range(1, layer_nodes[0] + 1)]
                influencer.plot_influence_score(influence_scores, feature_list,
                                            layer_name)
                                            
    
# Compare time performance between Shapley and Banzhaf
def experiment_01(model, X_train, sample_idx = 0):
    hl_1, hl_2, hl_3, prediction = model.predict(X_train)
    numb_layers = 4
    layer_names = ['input', 'h1', 'h2', 'h3']
    layer_nodes = [4, 8, 6, 4]
    layer_data = [X_train, hl_1, hl_2, hl_3]
    predict_list = {'input': model.predict_input,
                    'h1': model.predict_hl_1,
                    'h2': model.predict_hl_2,
                    'h3': model.predict_hl_3}

    with tf.Session(graph = model.graph) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model.CHECK_POINT)

        for idx in range(numb_layers):
            numb_features = layer_nodes[idx]
            layer_samples = layer_data[idx]
            feature_list = [str(i) for i in range(1, numb_features+1)]
            input_feature_list = [str(i) for i in range(1, layer_nodes[0] + 1)]
            influencer = QII(layer_samples, feature_list)
            X_individual = layer_samples[sample_idx]
        
            layer_name = layer_names[idx]
            predict_model = predict_list[layer_name]
                
            print ('Calculating Shapley...')
            start_time = time.time()        
            shapley_val = influencer.shapley(X_individual, predict_model, sess,
                                             is_exhaustive = True)
            print("--- %s seconds ---" % (time.time() - start_time))
            print (shapley_val)
                
            influence_scores = np.array([shapley_val[i] for i in feature_list])
            influence_scores = np.expand_dims(influence_scores, axis = 0)
               
            # if observed layer is not input, then we must map back
            if (idx != 0):
                influencer.plot_influence_score(influence_scores, feature_list,
                                             'Original Shapley ' + layer_name)
                weight_name_list = ['w1', 'w2', 'w3']
                weight_name_list = weight_name_list[:idx]
                influence_scores = influencer.mapback(model, X_train, influence_scores, weight_name_list)
                influencer.plot_influence_score(influence_scores, input_feature_list,
                                            'Mapback Shapley ' + layer_name)
            else:
                influencer.plot_influence_score(influence_scores, feature_list,
                                            'Shapley ' + layer_name)
                
            print ('Calculating Banzhaf...')
            start_time = time.time()        
            banzhaf_val = influencer.banzhaf(X_individual, predict_model, sess,
                                             is_exhaustive = True)
            print("--- %s seconds ---" % (time.time() - start_time))
            print (banzhaf_val)
                
            influence_scores = np.array([banzhaf_val[i] for i in feature_list])
            influence_scores = np.expand_dims(influence_scores, axis = 0)
                
            # if observed layer is not input, then we must map back
            if (idx != 0):
                influencer.plot_influence_score(influence_scores, feature_list,
                                            'Original Banzhaf ' + layer_name)
                weight_name_list = ['w1', 'w2', 'w3']
                weight_name_list = weight_name_list[:idx]
                influence_scores = influencer.mapback(model, X_train, influence_scores, weight_name_list)
                influencer.plot_influence_score(influence_scores, input_feature_list,
                                            'Mapback Banzhaf ' + layer_name)
            else:
                influencer.plot_influence_score(influence_scores, feature_list,
                                                'Banzhaf ' + layer_name)
 
# Test Shapley and Banzhaf value on selected data point: centroid and boundary
# along with specific set of features want to observe: 1 and 4
def experiment_02(model, X_train):
    hl_1, hl_2, hl_3, prediction = model.predict(X_train)
    numb_features = 4
    layer_name = 'input'
    
    # value of corresponding index in original data is mapped to X_train after splitted
    iris_setosa_centroid = 14
    iris_setosa_bound = 41
    iris_setosa = [iris_setosa_centroid, iris_setosa_bound]
    iris_versicolor_centroid = 27
    iris_versicolor_bound = [5, 47]
    iris_versicolor = [iris_versicolor_centroid] + iris_versicolor_bound 
    iris_virginica_centroid = 94
    iris_virginica_bound = 101
    iris_virginica = [iris_virginica_centroid, iris_virginica_bound]
    sample_idxs = iris_setosa + iris_versicolor + iris_virginica
    sample_names = ['setosa_centroid', 'setosa_bound',
                    'versicolor_centroid', 'versicolor_bound_1', 'versicolor_bound_2',
                    'virginica_centroid', 'virginica_bound']

    with tf.Session(graph = model.graph) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model.CHECK_POINT)

        for i in range(len(sample_idxs)):
            sample_idx = sample_idxs[i]
            sample_name = sample_names[i]
            print (sample_name)
            feature_list = [str(i) for i in range(1, numb_features+1)]
            observed_feature_list = [0, 3]
            influencer = QII(X_train, feature_list)
            X_individual = X_train[sample_idx]
            predict_model = model.predict_input
            shapley_val = influencer.shapley(X_individual, predict_model, sess,
                                                 observed_feature_list, is_exhaustive = True)                
            influence_scores = np.array([shapley_val[i] for i in feature_list])
            influence_scores = np.expand_dims(influence_scores, axis = 0)
            influencer.plot_influence_score(influence_scores, feature_list,
                                            'Shapley ' + layer_name + ' ' + sample_name)
                    
            banzhaf_val = influencer.banzhaf(X_individual, predict_model, sess,
                                             observed_feature_list, is_exhaustive = True)                
            influence_scores = np.array([banzhaf_val[i] for i in feature_list])
            influence_scores = np.expand_dims(influence_scores, axis = 0)
            influencer.plot_influence_score(influence_scores, feature_list,
                                            'Banzhaf ' + layer_name + ' ' + sample_name)

# What would happen if we focus on non-zero value in hidden layer 2
def experiment_03(model, train):  
    hl_1, hl_2, hl_3, prediction = model.predict(X_train)
    numb_features = 6
    layer_name = 'h2'
    
    # value of corresponding index in original data is mapped to X_train after splitted
    iris_setosa_centroid = 14
    iris_setosa_bound = 41
    iris_setosa = [iris_setosa_centroid, iris_setosa_bound]
    iris_versicolor_centroid = 27
    iris_versicolor_bound = [5, 47]
    iris_versicolor = [iris_versicolor_centroid] + iris_versicolor_bound 
    iris_virginica_centroid = 94
    iris_virginica_bound = 101
    iris_virginica = [iris_virginica_centroid, iris_virginica_bound]
    sample_idxs = iris_setosa + iris_versicolor + iris_virginica
    sample_names = ['setosa_centroid', 'setosa_bound',
                    'versicolor_centroid', 'versicolor_bound_1', 'versicolor_bound_2',
                    'virginica_centroid', 'virginica_bound']

    with tf.Session(graph = model.graph) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model.CHECK_POINT)

        for i in range(len(sample_idxs)):
            sample_idx = sample_idxs[i]
            sample_name = sample_names[i]
            print (sample_name)
            feature_list = [str(i) for i in range(1, numb_features+1)]
            input_feature_list = [str(i) for i in range(1, 4 + 1)]            
            observed_feature_list = [0, 2]
            influencer = QII(hl_2, feature_list)
            X_individual = hl_2[sample_idx]
            predict_model = model.predict_hl_2
            weight_name_list = ['w1', 'w2']
            shapley_val = influencer.shapley(X_individual, predict_model, sess,
                                                 observed_feature_list, is_exhaustive = True)                
            influence_scores = np.array([shapley_val[i] for i in feature_list])
            influence_scores = np.expand_dims(influence_scores, axis = 0)
            influencer.plot_influence_score(influence_scores, feature_list,
                                            'Original Shapley ' + layer_name + ' ' + sample_name)
            influence_scores = influencer.mapback(model, X_train, influence_scores, weight_name_list)
            influencer.plot_influence_score(influence_scores, input_feature_list,
                                            'Mapback Shapley ' + layer_name + ' ' + sample_name)
                    
            banzhaf_val = influencer.banzhaf(X_individual, predict_model, sess,
                                             observed_feature_list, is_exhaustive = True)                
            influence_scores = np.array([banzhaf_val[i] for i in feature_list])
            influence_scores = np.expand_dims(influence_scores, axis = 0)
            influencer.plot_influence_score(influence_scores, feature_list,
                                            'Original Banzhaf ' + layer_name + ' ' + sample_name)
            influence_scores = influencer.mapback(model, X_train, influence_scores, weight_name_list)
            influencer.plot_influence_score(influence_scores, input_feature_list,
                                            'Mapback Banzhaf ' + layer_name + ' ' + sample_name)
            
# What would happen if we focus on non-zero value in hidden layer 3
def experiment_04(model, train):  
    hl_1, hl_2, hl_3, prediction = model.predict(X_train)
    numb_features = 4
    layer_name = 'h3'
    
    # value of corresponding index in original data is mapped to X_train after splitted
    iris_setosa_centroid = 14
    iris_setosa_bound = 41
    iris_setosa = [iris_setosa_centroid, iris_setosa_bound]
    iris_versicolor_centroid = 27
    iris_versicolor_bound = [5, 47]
    iris_versicolor = [iris_versicolor_centroid] + iris_versicolor_bound 
    iris_virginica_centroid = 94
    iris_virginica_bound = 101
    iris_virginica = [iris_virginica_centroid, iris_virginica_bound]
    sample_idxs = iris_setosa + iris_versicolor + iris_virginica
    sample_names = ['setosa_centroid', 'setosa_bound',
                    'versicolor_centroid', 'versicolor_bound_1', 'versicolor_bound_2',
                    'virginica_centroid', 'virginica_bound']

    with tf.Session(graph = model.graph) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model.CHECK_POINT)

        for i in range(len(sample_idxs)):
            sample_idx = sample_idxs[i]
            sample_name = sample_names[i]
            print (sample_name)
            feature_list = [str(i) for i in range(1, numb_features+1)]
            input_feature_list = [str(i) for i in range(1, 4 + 1)]            
            observed_feature_list = [0, 1]
            influencer = QII(hl_3, feature_list)
            X_individual = hl_3[sample_idx]
            predict_model = model.predict_hl_3
            weight_name_list = ['w1', 'w2', 'w3']
            shapley_val = influencer.shapley(X_individual, predict_model, sess,
                                                 observed_feature_list, is_exhaustive = True)                
            influence_scores = np.array([shapley_val[i] for i in feature_list])
            influence_scores = np.expand_dims(influence_scores, axis = 0)
            influencer.plot_influence_score(influence_scores, feature_list,
                                            'Original Shapley ' + layer_name + ' ' + sample_name)
            influence_scores = influencer.mapback(model, X_train, influence_scores, weight_name_list)
            influencer.plot_influence_score(influence_scores, input_feature_list,
                                            'Mapback Shapley ' + layer_name + ' ' + sample_name)
                    
            banzhaf_val = influencer.banzhaf(X_individual, predict_model, sess,
                                             observed_feature_list, is_exhaustive = True)                
            influence_scores = np.array([banzhaf_val[i] for i in feature_list])
            influence_scores = np.expand_dims(influence_scores, axis = 0)
            influencer.plot_influence_score(influence_scores, feature_list,
                                            'Original Banzhaf ' + layer_name + ' ' + sample_name)
            influence_scores = influencer.mapback(model, X_train, influence_scores, weight_name_list)
            influencer.plot_influence_score(influence_scores, input_feature_list,
                                            'Mapback Banzhaf ' + layer_name + ' ' + sample_name)
            
# Compare evaluation value between exhaustive and approximate method
# on hidden layer 1
def experiment_05(model, X_train):
    hl_1, hl_2, hl_3, prediction = model.predict(X_train)
    numb_features = 8
    
    with tf.Session(graph = model.graph) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model.CHECK_POINT)

        feature_list = [str(i) for i in range(1, numb_features+1)]
        influencer = QII(hl_1, feature_list)
        X_individual = hl_1[27]
        predict_model = model.predict_hl_1
        print ('Calculating Approximate Shapley...')
        start_time = time.time()
        shapley_val = influencer.shapley(X_individual, predict_model, sess,
                                                 is_exhaustive = False)                
        print("--- %s seconds ---" % (time.time() - start_time))
        influence_scores = np.array([shapley_val[i] for i in feature_list])
        print (influence_scores)

        print ('Calculating Exhaustive Shapley...')
        start_time = time.time()
        shapley_val = influencer.shapley(X_individual, predict_model, sess,
                                                 is_exhaustive = True)                
        print("--- %s seconds ---" % (time.time() - start_time))
        influence_scores = np.array([shapley_val[i] for i in feature_list])
        print (influence_scores)

        
        
        print ('Calculating Approximate Banzhaf...')
        start_time = time.time()        
        banzhaf_val = influencer.banzhaf(X_individual, predict_model, sess,
                                             is_exhaustive = False)                
        print("--- %s seconds ---" % (time.time() - start_time))
        influence_scores = np.array([banzhaf_val[i] for i in feature_list])
        print (influence_scores)
    
        print ('Calculating Exhaustive Banzhaf...')
        start_time = time.time()
        banzhaf_val = influencer.banzhaf(X_individual, predict_model, sess,
                                                 is_exhaustive = True)                
        print("--- %s seconds ---" % (time.time() - start_time))
        influence_scores = np.array([banzhaf_val[i] for i in feature_list])
        print (influence_scores)

# Compare all quantity of interest, assignment, mapback on hidden layer 3
# datapoint = versicolor_centroid
def experiment_06(model, X_train):
    hl_1, hl_2, hl_3, prediction = model.predict(X_train)
    numb_features = 4 
    quantity_list = ['no_change_ratio', 'difference_ratio']
    assignment_list = ['normalized', 'ranking', 'nochange']
    propagate_list = ['transpose_mul', 'inverse_mul', 'softmax_dist',
                      'shift_dist', 'shift_softmax_dist']

    feature_list = [str(i) for i in range(1, numb_features+1)]
    input_feature_list = [str(i) for i in range(1, 4+1)]
    influencer = QII(hl_3, feature_list)
    X_individual = hl_3[27]
    predict_model = model.predict_hl_3

    with tf.Session(graph = model.graph) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model.CHECK_POINT)

        
        for quantity in quantity_list:
            shapley_val = influencer.shapley(X_individual, predict_model, sess,
                                             is_exhaustive = False, quantity=quantity)                
            influence_scores = np.array([shapley_val[i] for i in feature_list])
            influence_scores = np.expand_dims(influence_scores, axis = 0)
            weight_name_list = ['w1', 'w2', 'w3']
            influencer.plot_influence_score(influence_scores, feature_list,
                                            'Original ' + quantity)            
            for assignment in assignment_list:
                for propagate in propagate_list:
                    influence_scores = influencer.mapback(model, X_train, influence_scores, 
                                        weight_name_list, assigment=assignment, mapback=propagate)
                    influencer.plot_influence_score(influence_scores, input_feature_list,
                                            'Mapback '+assignment+' '+propagate)
            
                      
if __name__ == '__main__':
    # SET UP PARAMETERS
    set_up_parameter()
    
    # LOAD DATASET
    dataset = IRISDataset()
    X_test, X_train, y_test, y_train = dataset.split_to_train_test(seed=10)
    
    # LOAD/TRAIN MODEL
    model = NeuralNetwork()
    if (SHOULD_REUSE_MODEL == False):
        model.initialize_variables()
        model.create_graph()
        model.train(X_train, y_train, PRINT_STEPS, EPOCH)
        model.test(X_test, y_test)
    else:
        model.create_graph()
        
    # EXPERIMENTS
    #experiment_01(model, X_train)
    #experiment_02(model, X_train)
    #experiment_03(model, X_train)
    #experiment_04(model, X_train)
    #experiment_05(model, X_train)
    experiment_06(model, X_train)
    #calculate_shapley(model, X_train, iris_sentosa_centroid)
    