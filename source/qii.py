from iris_dataset import IRISDataset
from model import NeuralNetwork
import numpy as np
import math
import tensorflow as tf
import itertools
import seaborn as sns
import matplotlib.pyplot as plt

class QII(object):
    def __init__(self, X_train, feature_list):
        '''
        @params:
            - X_train: all data used for training model
            - feature_list: list of name of features
        '''
        self._X_train = X_train
        self._feature_list = feature_list
        self._quantity_list = {
                               'no_change_ratio': self.calculate_nochange_ratio,
                               'change_ratio': self.calculate_change_ratio,
                               'difference_ratio': self.calculate_difference_ratio}
        self._assignment_list = {
                                 'normalized': self.normalize_assigment,
                                 'ranking': self.ranking_assigment,
                                 'nochange': self.nochange_assignment}
        self._propagate_list = {
                                'transpose_mul': self.transpose_mul_propagate,
                                'inverse_mul': self.inverse_mul_propagate,
                                'softmax_dist': self.softmax_distribute_propagate,
                                'shift_dist': self.shift_distribute_propagate,
                                'shift_softmax_dist': self.shift_softmax_distribute_propagate}
            
    ### QUANTITIES OF INTEREST ###
    def calculate_nochange_ratio(self, X, X_pool, y_0, feature_inds, predict, sess):
        X_rep = np.tile(X, (self._pool_samples, 1))
        for feature_ind in feature_inds:
            X_rep[:, feature_ind] = X_pool[:, feature_ind]
        prediction = predict(X_rep, sess)
        nochange_ratio = np.mean(np.equal(np.argmax(prediction, axis=1), np.argmax(y_0, axis=1)))
        return nochange_ratio
        
    def calculate_change_ratio(self, X, X_pool, y_0, feature_inds, predict, sess):
        return 1.0 - self.calculate_nochange_ratio(X, X_pool, y_0, feature_inds, predict, sess)
        
    
    def calculate_difference_ratio(self, X, X_pool, y_0, feature_inds, predict, sess):
        X_rep = np.tile(X, (self._pool_samples, 1))
        for feature_ind in feature_inds:
            X_rep[:, feature_ind] = X_pool[:, feature_ind]
        prediction = predict(X_rep, sess)
        difference_ratio = np.mean(np.sum(np.abs(prediction - y_0)))
        return difference_ratio
        
        
    def shapley(self, x_individual, predict, sess, observed_feature_list = None, is_exhaustive = False,
                pool_samples = 600, numb_samplings = 600, quantity='no_change_ratio'):
        '''
        Calculate shapley value
        @params:
            - x_individual: datapoint need to be checked
            - predict: predict function of any model built-in Tensorflow that took
            datapoints and session
            - sess: session used to run predict
            - observed_feature_list: integer list of feature want to evaluate, start from 0
            - is_exhaustive: calculate exact/approximation Shapley value
            - pool_samples: number of data used for calculating quantity of interest
            - numb_samplings: number of samplings in case want to calculate approximation
            - quantity: name of quantity of interest
        @returns:
            - shapley value: dictionary contains shapley for each feature
        '''
        # Predefine sampling parameters
        if (observed_feature_list == None):
            observed_feature_list = [int(i)-1 for i in self._feature_list]
        numb_features = len(observed_feature_list)  

        if (is_exhaustive == True):
            self._pool_samples = len(self._X_train)
            self._numb_samplings = math.factorial(numb_features)
        else:
            self._pool_samples = min(pool_samples, len(self._X_train))
            self._numb_samplings = min(numb_samplings, math.factorial(numb_features))
                        
        y0 = predict(np.expand_dims(x_individual, axis = 0), sess)
        b = np.random.randint(0, self._X_train.shape[0], self._pool_samples)
        X_pool = self._X_train[b]
    
        shapley = dict.fromkeys(self._feature_list, 0)
    
        # Calculate Shapley value...
        if (is_exhaustive == True):
            # exactly
            permutations = list(itertools.permutations(observed_feature_list))
            for sample in permutations:
                perm = sample
                
                for i in range(0, numb_features):
                    # Choose the set of feature standing before i in the permutation
                    si_ind = perm[i]
                    si = self._feature_list[si_ind]
                    S = [perm[j] for j in range(0, i)]
                    p_S = self._quantity_list[quantity](x_individual, X_pool, y0, S, predict, sess)
                    #also intervene on si ~ replace its value with random values in pool
                    p_S_si = self._quantity_list[quantity](x_individual, X_pool, y0, S + [si_ind], predict, sess)
                    shapley[si] = shapley[si] - (p_S_si - p_S)
        else:
            # or approximately
            for sample in range(0, self._numb_samplings):
                perm = np.random.permutation(len(self._feature_list))
            
                for i in range(0, numb_features):
                    si_ind = perm[i]
                    si = self._feature_list[si_ind]
                    S = [perm[j] for j in range(0, i)]
                    p_S = self._quantity_list[quantity](x_individual, X_pool, y0, S, predict, sess)
                    p_S_si = self._quantity_list[quantity](x_individual, X_pool, y0, S + [si_ind], predict, sess)
                    shapley[si] = shapley[si] - (p_S_si - p_S)
        
        for feature in shapley:
            shapley[feature] = shapley[feature]/self._numb_samplings
        return shapley

    def banzhaf(self, x_individual, predict, sess, observed_feature_list = None, is_exhaustive = False,
                pool_samples = 600, numb_samplings = 600, quantity='no_change_ratio'):
        '''
        Calculate banzhaf value
        @param:
            - x_individual: datapoint need to be checked
            - predict: predict function of any model built-in Tensorflow that took
            datapoints and session
            - sess: session used to run predict
            - observed_feature_list: integer list of feature want to evaluate, start from 0
            - is_exhaustive: calculate exact Shapley value
            - pool_samples: number of data used for calculating quantity of interest
            - numb_samplings: number of samplings in case want to calculate approximation
            - quantity: name of quantity of interest
        @returns:
            - banzhaf value: dictionary contains banzhaf for each feature
        '''
        
        # Predefine sampling parameters
        if (observed_feature_list == None):
            observed_feature_list = [int(i)-1 for i in self._feature_list]
        numb_features = len(observed_feature_list)   
        
        if (is_exhaustive == True):
            self._pool_samples = len(self._X_train)
            self._numb_samplings = math.factorial(numb_features)
        else:
            self._pool_samples = min(pool_samples, len(self._X_train))
            self._numb_samplings = min(numb_samplings, math.factorial(numb_features))
        
        def get_all_combinations(feature_inds):
            numb_features = len(feature_inds)
            combinations = []
            for subset_size in range(1, numb_features+1):
                combination_subset_i = list(itertools.combinations(feature_inds, subset_size))
                combinations = combinations + combination_subset_i
            return combinations
            
        y0 = predict(np.expand_dims(x_individual, axis = 0), sess)
        b = np.random.randint(0, self._X_train.shape[0], self._pool_samples)
        X_pool = self._X_train[b]

        banzhaf = dict.fromkeys(self._feature_list, 0)
        
        # Calculate Banzhaf value...
        if (is_exhaustive == True):
            # exactly
            combinations = get_all_combinations(observed_feature_list)
            
            for sample in combinations:
                subset = sample
                for si_ind in subset:
                    # Chooses the set of feature in observed subset, excluding i
                    si = self._feature_list[si_ind]
                    S = [j for j in subset if j != si_ind]
                    p_S = self._quantity_list[quantity](x_individual, X_pool, y0, S, predict, sess)
                    # also intervenes on s_i
                    p_S_si = self._quantity_list[quantity](x_individual, X_pool, y0, S + [si_ind], predict, sess)
                    banzhaf[si] = banzhaf[si] - (p_S_si - p_S)
        else:
            # or approximately
            for sample in range(0, self._numb_samplings):
                r = np.random.ranf(numb_features)
                subset = [i for i in range(numb_features) if r[i] > 0.5]
                
                for si_ind in range(0, numb_features):
                    si = self._feature_list[si_ind]
                    S = [j for j in subset if j != si_ind]
                    p_S = self._quantity_list[quantity](x_individual, X_pool, y0, S, predict, sess)
                    p_S_si = self._quantity_list[quantity](x_individual, X_pool, y0, S + [si_ind], predict, sess)
                    banzhaf[si] = banzhaf[si] - (p_S_si - p_S)

        for feature in banzhaf:
            banzhaf[feature] = banzhaf[feature]/self._numb_samplings
        return banzhaf

    ### REPRESENTATION OF INFLUENCE SCORE ###
    def ranking_assigment(self, influence_scores):
        '''
        @param:
            - influence_score: numpy array 1D
        @return:
            - ranking assigments
        '''
        sort_index = np.argsort(influence_scores)
        return sort_index
        
    def normalize_assigment(self, influence_scores):
        '''
        @param:
            - influence_score: numpy array 1D
        @return:
            - distribute value in such a way that total sum is 1.0
            and every element is non-negative
        '''
        ex = np.exp(influence_scores)
        softmax = ex / ex.sum()
        return softmax
        
    def nochange_assignment(self, influence_scores):
        return influence_scores
        
    ### MAPPING BACK TECHNIQUES ###
    def mapback(self, model, X_train, influence_scores, weight_name_list,
                assigment = 'nochange', mapback = 'transpose_mul'):
        '''
        Map back influence score from intermediate layer to input
        @params:
            - model: NeuralNetwork class has been trained
            - X_train
            - influence_scores: numpy array of Shapley/Banzhaf value of intermediate layer
            - assigment: assignment type
            - weight_name_list: weights extracted from model is dictionary, hence
            we need a chornological list of name from input to current observed layer
        @return:
            - input_influence_scores: influence score in input
        '''
        weights, biases = model.extract_weights_biases(X_train)
        assignment_scores = self._assignment_list[assigment](influence_scores)
        iterations = len(weight_name_list)
        assignment_scores = self._propagate_list[mapback](iterations, weights,
                                                         weight_name_list, assignment_scores)
        return assignment_scores
        
    def transpose_mul_propagate(self, iterations, weights, weight_name_list,
                                assignment_scores):
        for i in range(iterations-1, -1, -1):
            w_name = weight_name_list[i]
            w = weights[w_name]
            w_T = np.transpose(w)
            assignment_scores = np.dot(assignment_scores, w_T)
        return assignment_scores   
        
    def inverse_mul_propagate(self, iterations, weights, weight_name_list,
                                assignment_scores):
        for i in range(iterations-1, -1, -1):
            w_name = weight_name_list[i]
            w = weights[w_name]
            w_pinv = np.linalg.pinv(w)
            assignment_scores = np.dot(assignment_scores, w_pinv)
        return assignment_scores   
        
    def softmax_distribute_propagate(self, iterations, weights, weight_name_list,
                                assignment_scores):
        for i in range(iterations-1, -1, -1):
            w_name = weight_name_list[i]
            w = weights[w_name]
            w = np.exp(w)
            w_sum = w.sum(axis=1)
            for j in range(w.shape[0]):
                w[j] = w[j]/w_sum[j]
            w_T = np.transpose(w)
            assignment_scores = np.dot(assignment_scores, w_T)
        return assignment_scores 
        
    def shift_distribute_propagate(self, iterations, weights, weight_name_list,
                                assignment_scores):
        # shift to direction that make smallest value = 0
        for i in range(iterations-1, -1, -1):
            w_name = weight_name_list[i]
            w = weights[w_name]
            w_min = np.amin(w, axis = 1)
            for j in range(w.shape[0]):
                w[j] = w[j] - w_min[j]
            w_T = np.transpose(w)
            assignment_scores = np.dot(assignment_scores, w_T)
        return assignment_scores
        
    def shift_softmax_distribute_propagate(self, iterations, weights, weight_name_list,
                                assignment_scores):
        # shift to direction that make smallest value = 0
        # then do softmax
        for i in range(iterations-1, -1, -1):
            w_name = weight_name_list[i]
            w = weights[w_name]
            # shift
            w_min = np.amin(w, axis = 1)
            for j in range(w.shape[0]):
                w[j] = w[j] - w_min[j]
            # then softmax
            w = np.exp(w)
            w_sum = w.sum(axis=1)
            for j in range(w.shape[0]):
                w[j] = w[j]/w_sum[j]
            w_T = np.transpose(w)
            assignment_scores = np.dot(assignment_scores, w_T)
        return assignment_scores
    
    ### PLOT INFLUENCE SCORES ###
    def plot_influence_score(self, influence_scores, feature_list,
                             name = 'Figure of influence scores'):
        sns.set(style="white", context="talk")
        fig = plt.figure()
        x = np.array(feature_list)
        y = influence_scores[0]
        sns.barplot(x, y, palette="BuGn_d")
        plt.xlabel('Observed units')
        plt.ylabel('Influence scores')
        plt.title(name)
        fig = plt.plot()
        
def proper_use_shapley():
    def read_dataset():
        dataset = IRISDataset()
        X_test, X_train, y_test, y_train = dataset.split_to_train_test(seed=10)
        return X_test, X_train, y_test, y_train
        
    def train_model(X_test, X_train, y_test, y_train):
        print_steps = 50; epoch = 100
        model = NeuralNetwork()
        model.initialize_variables()
        model.create_graph()
        model.train(X_train, y_train, print_steps, epoch)
        model.test(X_test, y_test)      
        return model        
        
    def use_QII_with_model(X_train):
        model = NeuralNetwork()
        model.create_graph()
        
        # get input data for each layer
        hl_1, hl_2, hl_3, prediction = model.predict(X_train)
        
        # PREDEFINE PARAMETERS
        # Number of features available for sampling, we want to observe
        # input layer, hence number of units is 4
        numb_features = 4 
                
        # initializes QII class with name for each features, here we use index only
        feature_list = [str(i) for i in range(1, numb_features+1)]
        influencer = QII(X_train, feature_list)
    
        # value of one data points we want to observe
        data_point = X_train[0]
        
        with tf.Session(graph = model.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, model.CHECK_POINT)
        
            # user can choose to calculate shapley by random sampling of permutation
            # or exhaustively check each of them with is_exhaustive
            
            # since 4! = 24 is not a big number, we can try both here 
            print ('Approximation with Random Sampling')
            shapley_val = influencer.shapley(data_point, model.predict_input, sess,
                                             is_exhaustive = False)
            print (shapley_val)
            
            print ('Correct value with Exhausive Computation')
            shapley_val = influencer.shapley(data_point, model.predict_input, sess,
                                             is_exhaustive = True)
            print (shapley_val)
                
    X_test, X_train, y_test, y_train = read_dataset()
    train_model(X_test, X_train, y_test, y_train)
    use_QII_with_model(X_train)    

    
def proper_use_banzhaf():
    def read_dataset():
        dataset = IRISDataset()
        X_test, X_train, y_test, y_train = dataset.split_to_train_test(seed=10)
        return X_test, X_train, y_test, y_train
        
    def train_model(X_test, X_train, y_test, y_train):
        print_steps = 50; epoch = 100
        model = NeuralNetwork()
        model.initialize_variables()
        model.create_graph()
        model.train(X_train, y_train, print_steps, epoch)
        model.test(X_test, y_test)      
        return model        
        
    def use_QII_with_model(X_train):
        model = NeuralNetwork()
        model.create_graph()
        
        # get input data for each layer
        hl_1, hl_2, hl_3, prediction = model.predict(X_train)
        
        # Number of features available for sampling, we want to observe
        # input layer, hence number of units is 4
        numb_features = 4 
             
        # initializes QII class with name for each features, here we use index only
        feature_list = [str(i) for i in range(1, numb_features+1)]
        influencer = QII(X_train, feature_list)
    
        # value of one data points we want to observe
        data_point = X_train[0]
        
        with tf.Session(graph = model.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, model.CHECK_POINT)
        
            # user can choose to calculate shapley by random sampling of subsets
            # or exhaustively check each of them with is_exhaustive
            
            # since 4! = 24 is not a big number, we can try both here 
            print ('Approximation with Random Sampling')
            banzhaf_val = influencer.banzhaf(data_point, model.predict_input, sess,
                                             is_exhaustive = False)
            print (banzhaf_val)
            
            print ('Correct value with Exhaustive Computation')
            banzhaf_val = influencer.banzhaf(data_point, model.predict_input, sess,
                                             is_exhaustive = True)
            print (banzhaf_val)
            
    X_test, X_train, y_test, y_train = read_dataset()
    train_model(X_test, X_train, y_test, y_train)
    use_QII_with_model(X_train)    

def proper_use_mapback_shapley():
    def read_dataset():
        dataset = IRISDataset()
        X_test, X_train, y_test, y_train = dataset.split_to_train_test(seed=10)
        return X_test, X_train, y_test, y_train
        
    def train_model(X_test, X_train, y_test, y_train):
        model = NeuralNetwork()
        model.create_graph()
        return model        
        
    def use_QII_with_model(X_train):
        model = NeuralNetwork()
        model.create_graph()
        hl_1, hl_2, hl_3, prediction = model.predict(X_train)
        numb_features = 4 # hidden layer 3 has 4 units
        feature_list = [str(i) for i in range(1, numb_features+1)]
        influencer = QII(hl_3, feature_list)
    
        # value of one data points we want to observe
        data_point = hl_3[0]
        
        with tf.Session(graph = model.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, model.CHECK_POINT)
            print ('Correct value with Exhaustive Computation')
            shapley_val = influencer.shapley(data_point, model.predict_hl_3, sess,
                                             is_exhaustive = True)
            print (shapley_val)
        
            influence_scores = np.array([shapley_val[i] for i in feature_list])
            influence_scores = np.expand_dims(influence_scores, axis = 0)
            weight_name_list = ['w1', 'w2', 'w3']
            influence_scores = influencer.mapback(model, X_train, influence_scores, weight_name_list)
            influencer.plot_influence_score(influence_scores, feature_list)
            
    X_test, X_train, y_test, y_train = read_dataset()
    train_model(X_test, X_train, y_test, y_train)
    use_QII_with_model(X_train)    

def proper_use_mapback_banzhaf():
    def read_dataset():
        dataset = IRISDataset()
        X_test, X_train, y_test, y_train = dataset.split_to_train_test(seed=10)
        return X_test, X_train, y_test, y_train
        
    def train_model(X_test, X_train, y_test, y_train):
        print_steps = 50; epoch = 100
        model = NeuralNetwork()
        model.initialize_variables()
        model.create_graph()
        model.train(X_train, y_train, print_steps, epoch)
        model.test(X_test, y_test)      
        return model        
        
    def use_QII_with_model(X_train):
        model = NeuralNetwork()
        model.create_graph()
        hl_1, hl_2, hl_3, prediction = model.predict(X_train)
        numb_features = 4 # hidden layer 3 has 4 units
        feature_list = [str(i) for i in range(1, numb_features+1)]
        influencer = QII(hl_3, feature_list)
    
        # value of one data points we want to observe
        data_point = hl_3[0]
        
        with tf.Session(graph = model.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, model.CHECK_POINT)
            print ('Correct value with Exhaustive Computation')
            banzhaf_val = influencer.banzhaf(data_point, model.predict_hl_3, sess,
                                             is_exhaustive = True)
            print (banzhaf_val)
        
            influence_scores = np.array([banzhaf_val[i] for i in feature_list])
            influence_scores = np.expand_dims(influence_scores, axis = 0)
            weight_name_list = ['w1', 'w2', 'w3']
            influence_scores = influencer.mapback(model, X_train, influence_scores, weight_name_list)
            influencer.plot_influence_score(influence_scores, feature_list)
            
    X_test, X_train, y_test, y_train = read_dataset()
    train_model(X_test, X_train, y_test, y_train)
    use_QII_with_model(X_train)
    
# Checks how long it takes to run 8 features Shapley
def test_case_1():
    import time
    dataset = IRISDataset()
    X_test, X_train, y_test, y_train = dataset.split_to_train_test(seed=10)
    model = NeuralNetwork()
    model.create_graph()
    hl_1, hl_2, hl_3, prediction = model.predict(X_train)
    numb_features = 8
    feature_list = [str(i) for i in range(1, numb_features+1)]
    influencer = QII(hl_1, feature_list)
    data_point = hl_1[0]
    with tf.Session(graph = model.graph) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model.CHECK_POINT)
        start_time = time.time()
        print ('Approximation with Random Sampling')
        shapley_val = influencer.shapley(data_point, model.predict_hl_1, sess,
                                         is_exhaustive = False)
        print (shapley_val)
        print("--- %s seconds ---" % (time.time() - start_time))
        
        start_time = time.time()
        print ('Correct value with Exhaustive Computation')
        shapley_val = influencer.shapley(data_point, model.predict_hl_1, sess,
                                             is_exhaustive = True)
        print (shapley_val)
        print("--- %s seconds ---" % (time.time() - start_time))
        
# Checks how long it takes to run 8 features Banzhaf        
def test_case_2():
    import time
    dataset = IRISDataset()
    X_test, X_train, y_test, y_train = dataset.split_to_train_test(seed=10)
    model = NeuralNetwork()
    model.create_graph()
    hl_1, hl_2, hl_3, prediction = model.predict(X_train)
    numb_features = 8
    feature_list = [str(i) for i in range(1, numb_features+1)]
    influencer = QII(hl_1, feature_list)
    data_point = hl_1[0]
    with tf.Session(graph = model.graph) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model.CHECK_POINT)
        start_time = time.time()
        print ('Approximation with Random Sampling')
        shapley_val = influencer.banzhaf(data_point, model.predict_hl_1, sess,
                                             is_exhaustive = False)
        print (shapley_val)
        print("--- %s seconds ---" % (time.time() - start_time))
        
        start_time = time.time()
        print ('Correct value with Exhaustive Computation')
        shapley_val = influencer.banzhaf(data_point, model.predict_hl_1, sess,
                                             is_exhaustive = True)
        print (shapley_val)
        print("--- %s seconds ---" % (time.time() - start_time))

# Test all map back Shapley
def test_case_3():
    def read_dataset():
        dataset = IRISDataset()
        X_test, X_train, y_test, y_train = dataset.split_to_train_test(seed=10)
        return X_test, X_train, y_test, y_train
                
    def use_QII_with_model(X_train):
        model = NeuralNetwork()
        model.create_graph()
        hl_1, hl_2, hl_3, prediction = model.predict(X_train)
        numb_features = 4 # hidden layer 3 has 4 units
        feature_list = [str(i) for i in range(1, numb_features+1)]
        influencer = QII(hl_3, feature_list)
    
        # value of one data points we want to observe
        data_point = hl_3[0]
        
        assignment_list = ['normalized', 'ranking', 'nochange']
        propagate_list = ['transpose_mul', 'inverse_mul', 'softmax_dist',
        'shift_dist', 'shift_softmax_dist']


        with tf.Session(graph = model.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, model.CHECK_POINT)
            for assignment in assignment_list:
                for propagate in propagate_list:
                    shapley_val = influencer.shapley(data_point, model.predict_hl_3, sess,
                                                     is_exhaustive = False)
                    influence_scores = np.array([shapley_val[i] for i in feature_list])
                    influence_scores = np.expand_dims(influence_scores, axis = 0)
                    weight_name_list = ['w1', 'w2', 'w3']
                    influence_scores = influencer.mapback(model, X_train, influence_scores, weight_name_list,
                                                          assignment, propagate)
                    influencer.plot_influence_score(influence_scores, feature_list)
                
    X_test, X_train, y_test, y_train = read_dataset()
    use_QII_with_model(X_train) 
    
# Test all map back Banzhaf
def test_case_4():
    def read_dataset():
        dataset = IRISDataset()
        X_test, X_train, y_test, y_train = dataset.split_to_train_test(seed=10)
        return X_test, X_train, y_test, y_train
                
    def use_QII_with_model(X_train):
        model = NeuralNetwork()
        model.create_graph()
        hl_1, hl_2, hl_3, prediction = model.predict(X_train)
        numb_features = 4 # hidden layer 3 has 4 units
        feature_list = [str(i) for i in range(1, numb_features+1)]
        influencer = QII(hl_3, feature_list)
    
        # value of one data points we want to observe
        data_point = hl_3[0]
        
        assignment_list = ['normalized', 'ranking', 'nochange']
        propagate_list = ['transpose_mul', 'inverse_mul', 'softmax_dist',
        'shift_dist', 'shift_softmax_dist']


        with tf.Session(graph = model.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, model.CHECK_POINT)
            for assignment in assignment_list:
                for propagate in propagate_list:
                    banzhaf_val = influencer.banzhaf(data_point, model.predict_hl_3, sess,
                                                     is_exhaustive = False)
                    influence_scores = np.array([banzhaf_val[i] for i in feature_list])
                    influence_scores = np.expand_dims(influence_scores, axis = 0)
                    weight_name_list = ['w1', 'w2', 'w3']
                    influence_scores = influencer.mapback(model, X_train, influence_scores, weight_name_list,
                                                          assignment, propagate)
                    influencer.plot_influence_score(influence_scores, feature_list)
                
    X_test, X_train, y_test, y_train = read_dataset()
    use_QII_with_model(X_train)
   
# Test 3 features out of 4 Shapley
def test_case_5():
    def read_dataset():
        dataset = IRISDataset()
        X_test, X_train, y_test, y_train = dataset.split_to_train_test(seed=10)
        return X_test, X_train, y_test, y_train
                
    def use_QII_with_model(X_train):
        model = NeuralNetwork()
        model.create_graph()
        hl_1, hl_2, hl_3, prediction = model.predict(X_train)
        numb_features = 6 # hidden layer 3 has 4 units
        feature_list = [str(i) for i in range(1, numb_features+1)]
        influencer = QII(hl_2, feature_list)
    
        # value of one data points we want to observe
        data_point = X_train[101]

        with tf.Session(graph = model.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, model.CHECK_POINT)
            observed_feature_list = [1, 3, 2]
            shapley_val = influencer.shapley(data_point, model.predict_input, sess,
                                             observed_feature_list, is_exhaustive = True)
            influence_scores = np.array([shapley_val[i] for i in feature_list])
            influence_scores = np.expand_dims(influence_scores, axis = 0)
            influencer.plot_influence_score(influence_scores, feature_list)
            
                
    X_test, X_train, y_test, y_train = read_dataset()
    use_QII_with_model(X_train)

# Test 3 features out of 4 Banzhaf
def test_case_6():
    def read_dataset():
        dataset = IRISDataset()
        X_test, X_train, y_test, y_train = dataset.split_to_train_test(seed=10)
        return X_test, X_train, y_test, y_train
                
    def use_QII_with_model(X_train):
        model = NeuralNetwork()
        model.create_graph()
        hl_1, hl_2, hl_3, prediction = model.predict(X_train)
        numb_features = 4 # hidden layer 3 has 4 units
        feature_list = [str(i) for i in range(1, numb_features+1)]
        influencer = QII(X_train, feature_list)
    
        # value of one data points we want to observe
        data_point = X_train[101]

        with tf.Session(graph = model.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, model.CHECK_POINT)
            observed_feature_list = [0, 1, 3]
            banzhaf_val = influencer.banzhaf(data_point, model.predict_input, sess,
                                             observed_feature_list, is_exhaustive = True)
            influence_scores = np.array([banzhaf_val[i] for i in feature_list])
            influence_scores = np.expand_dims(influence_scores, axis = 0)
            influencer.plot_influence_score(influence_scores, feature_list)
            
                
    X_test, X_train, y_test, y_train = read_dataset()
    use_QII_with_model(X_train)

    

if __name__ == '__main__':
    #proper_use_shapley()
    #proper_use_banzhaf()
    #proper_use_mapback_shapley()
    #proper_use_mapback_banzhaf()
    #test_case_1()
    #test_case_2()
    test_case_3()
    test_case_4()
    #test_case_5()
    #test_case_6()