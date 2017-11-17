from iris_dataset import IRISDataset
from model import NeuralNetwork
import numpy as np
import math
import tensorflow as tf
import os
import itertools

class QII(object):
    def __init__(self, pool_samples, numb_samplings, X_train, feature_list):
        self._pool_samples = pool_samples
        self._numb_samplings = numb_samplings
        self._X_train = X_train
        self._feature_list = feature_list
        
    def shapley(self, x_individual, predict, sess, is_exhaustive = False):
        
        if (self._pool_samples > 600):
            self._pool_samples = 600
        if (self._numb_samplings > 600):
            self._numb_samplings = 600
       
        def calculate_nochange_ratio(X, X_pool, y_0, feature_inds, predict, sess):
            X_rep = np.tile(X, (self._pool_samples, 1))
            for feature_ind in feature_inds:
                X_rep[:, feature_ind] = X_pool[:, feature_ind]
            prediction = predict(X_rep, sess)
            nochange_ratio = np.mean(np.equal(np.argmax(prediction, axis=1), np.argmax(y_0, axis=1)))
            return nochange_ratio
            
        y0 = predict(np.expand_dims(x_individual, axis = 0), sess)
        b = np.random.randint(0, self._X_train.shape[0], self._pool_samples)
        X_pool = self._X_train[b]
    
        shapley = dict.fromkeys(self._feature_list, 0)
    
        if (is_exhaustive == True):
            numb_features = len(self._feature_list)
            for sample in itertools.permutations(range(numb_features)):
                perm = sample
                #print ('Sampling' + str(sample))
                for i in range(0, len(self._feature_list)):
                    # Choose a random subset and get string indices by flattening
                    #  excluding si
                    si_ind = perm[i]
                    si = self._feature_list[si_ind]
                    S = [perm[j] for j in range(0, i)]
                    p_S = calculate_nochange_ratio(x_individual, X_pool, y0, S, predict, sess)
                    #also intervene on s_i
                    p_S_si = calculate_nochange_ratio(x_individual, X_pool, y0, S + [si_ind], predict, sess)
                    shapley[si] = shapley[si] - (p_S_si - p_S)/self._numb_samplings
        else:
            for sample in range(0, self._numb_samplings):
                perm = np.random.permutation(len(self._feature_list))
                #print ('Sampling' + str(sample))
                for i in range(0, len(self._feature_list)):
                    # Choose a random subset and get string indices by flattening
                    #  excluding si
                    si_ind = perm[i]
                    si = self._feature_list[si_ind]
                    S = [perm[j] for j in range(0, i)]
                    p_S = calculate_nochange_ratio(x_individual, X_pool, y0, S, predict, sess)
                    #also intervene on s_i
                    p_S_si = calculate_nochange_ratio(x_individual, X_pool, y0, S + [si_ind], predict, sess)
                    shapley[si] = shapley[si] - (p_S_si - p_S)/self._numb_samplings
        
        return shapley

    def banzhaf(self):
        pass
    
def test_case_1():
    print_steps = 50; epoch = 500
    dataset = IRISDataset()
    X_test, X_train, y_test, y_train = dataset.split_to_train_test(seed=10)
    
    model = NeuralNetwork()
    model.train(X_train, y_train, print_steps, epoch)
    model.test(X_test, y_test)

    numb_features = len(X_train[0]); pool_samples = len(X_train)
    numb_samplings = math.factorial(numb_features)
    
    feature_list = list(dataset._dataset.columns.values)[1:-1]
    influencer = QII(pool_samples, numb_samplings, X_train, feature_list)    
    
    with tf.Session(graph = model.graph) as sess:
        new_saver = tf.train.import_meta_graph(os.path.join(os.getcwd(),
                                                                'saved_model_01.meta'))
        new_saver.restore(sess, tf.train.latest_checkpoint('./'))
        shapley_val = influencer.shapley(X_train[0], model.predict_input, sess,
                                         is_exhaustive = True)
    print (shapley_val)
    
def test_case_2():
    print_steps = 50; epoch = 500
    dataset = IRISDataset()
    X_test, X_train, y_test, y_train = dataset.split_to_train_test(seed=10)
    
    model = NeuralNetwork()
    model.train(X_train, y_train, print_steps, epoch)
    hl_1, hl_2, hl_3, prediction = model.predict(X_train)
    
    numb_features = 8; pool_samples = len(X_train)
    numb_samplings = math.factorial(numb_features)
    
    feature_list = [str(i) for i in range(1, numb_features+1)]
    influencer = QII(pool_samples, numb_samplings, hl_1, feature_list)
    
    with tf.Session(graph = model.graph) as sess:
        new_saver = tf.train.import_meta_graph(os.path.join(os.getcwd(),
                                                                'saved_model_01.meta'))
        new_saver.restore(sess, tf.train.latest_checkpoint('./'))
        shapley_val = influencer.shapley(hl_1[0], model.predict_hl_1, sess,
                                         is_exhaustive = False)
    print (shapley_val)
    
if __name__ == '__main__':
    #test_case_1()
    test_case_2()