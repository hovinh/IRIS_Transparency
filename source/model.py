import tensorflow as tf
import os
from iris_dataset import IRISDataset

# This helper function taken from official TensorFlow documentation,
# simply add some ops that take care of logging summaries
def variable_summaries(var, name_scope):
    with tf.name_scope(name_scope):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
        
class NeuralNetwork(object):
    LOG_DIR = os.path.join(os.getcwd(), 'log')
    CHECK_POINT = 'saved_model'
    input_layer_nodes = 4
    output_layer_nodes = 3            
    hidden_layer_nodes = {'h1': 8, 'h2': 6, 'h3': 4}

    def __init__(self):
        super(NeuralNetwork, self).__init__()
    
    def initialize_variables(self):
        '''
        Initialize all required variables (weights, biases) therefore can be
        loaded in different session in mutiple different run
        '''
        self.init_graph = tf.Graph()
        
        # Define all required variables
        with self.init_graph.as_default():
            
            # Create variables for Neural Network layers
            with tf.variable_scope('variables'):
                weights = {
                'w1': tf.Variable(tf.random_normal(shape=[self.input_layer_nodes,
                                                          self.hidden_layer_nodes['h1']]),
                                                            name = 'W1'),
                'w2': tf.Variable(tf.random_normal(shape=[self.hidden_layer_nodes['h1'],
                                                          self.hidden_layer_nodes['h2']]),
                                                            name = 'W2'),
                'w3': tf.Variable(tf.random_normal(shape=[self.hidden_layer_nodes['h2'],
                                                          self.hidden_layer_nodes['h3']]),
                                                            name = 'W3'),
                'w4': tf.Variable(tf.random_normal(shape=[self.hidden_layer_nodes['h3'],
                                                          self.output_layer_nodes]),
                                                            name = 'W4')
                }
                
                biases = {
                'b1': tf.Variable(tf.random_normal(shape=[self.hidden_layer_nodes['h1']]),
                                  name = 'b1'),
                'b2': tf.Variable(tf.random_normal(shape=[self.hidden_layer_nodes['h2']]),
                                  name = 'b2'),
                'b3': tf.Variable(tf.random_normal(shape=[self.hidden_layer_nodes['h3']]),
                                  name = 'b3'),
                'b4': tf.Variable(tf.random_normal(shape=[self.output_layer_nodes]),
                                  name = 'b4')
                }
                
            # Initialize all variables
            self.init_op = tf.global_variables_initializer()
        
        with tf.Session(graph = self.init_graph) as sess:
            # Initialize variables
            sess.run(self.init_op)
            weight_list = [weights['w1'], weights['w2'], weights['w3'], weights['w4']]
            bias_list = [biases['b1'], biases['b2'], biases['b3'], biases['b4']] 
            saver = tf.train.Saver(weight_list + bias_list)
            saver.save(sess, os.path.join(os.getcwd(), self.CHECK_POINT))
            
    def create_graph(self):
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            
            # Initialize placeholders
            
            self.X_data = tf.placeholder(shape=[None, self.input_layer_nodes], 
                                         name='Input', dtype=tf.float32)
            self.y_target = tf.placeholder(shape=[None, self.output_layer_nodes],
                                         name='Labels', dtype=tf.float32)
            variable_summaries(self.X_data, 'Input')
            variable_summaries(self.y_target, 'Actual_Label')
            with tf.variable_scope('variables'):
                # Create variables for Neural Network layers
                self.weights = {
                'w1': tf.Variable(-1.0, validate_shape=False, name = 'W1'),             
                'w2': tf.Variable(-1.0, validate_shape=False, name = 'W2'),
                'w3': tf.Variable(-1.0, validate_shape=False, name = 'W3'),
                'w4': tf.Variable(-1.0, validate_shape=False, name = 'W4')
                }
                
                variable_summaries(self.weights['w1'], 'Weight_Input_Hidden1')
                variable_summaries(self.weights['w2'], 'Weight_Hidden1_Hidden2')
                variable_summaries(self.weights['w3'], 'Weight_Hidden2_Hidden3')
                variable_summaries(self.weights['w4'], 'Weight_Hidden3_Output')
                
                self.biases = {
                'b1': tf.Variable(-1.0, validate_shape=False, name = 'b1'),
                'b2': tf.Variable(-1.0, validate_shape=False, name = 'b2'),
                'b3': tf.Variable(-1.0, validate_shape=False, name = 'b3'),
                'b4': tf.Variable(-1.0, validate_shape=False, name = 'b4')
                }
                variable_summaries(self.biases['b1'], 'Biases_Input_Hidden1')
                variable_summaries(self.biases['b2'], 'Biases_Hidden1_Hidden2')
                variable_summaries(self.biases['b3'], 'Biases_Hidden2_Hidden3')
                variable_summaries(self.biases['b4'], 'Biases_Hidden3_Hidden4')
    
            with tf.variable_scope('model'):
                # Operations
                self.hidden_output_1 = tf.nn.relu(tf.matmul(self.X_data, self.weights['w1']) + self.biases['b1'],
                                             name = 'Hidden_1')
                self.hidden_output_2 = tf.nn.relu(tf.matmul(self.hidden_output_1, self.weights['w2']) + self.biases['b2'],
                                             name = 'Hidden_2')
                self.hidden_output_3 = tf.nn.relu(tf.matmul(self.hidden_output_2, self.weights['w3']) + self.biases['b3'],
                                             name = 'Hidden_3')
                self.raw_output = tf.matmul(self.hidden_output_3, self.weights['w4']) + self.biases['b4']
                self.final_output = tf.nn.softmax(self.raw_output,
                                             name = 'Output')
                variable_summaries(self.hidden_output_1, 'Hidden_Layer_01')
                variable_summaries(self.hidden_output_2, 'Hidden_Layer_02')
                variable_summaries(self.hidden_output_3, 'Hidden_Layer_03')
                variable_summaries(self.final_output, 'Output')                                  

                # Cost Function
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.raw_output, labels=self.y_target))
                tf.summary.scalar('loss', self.loss)
                
                # Optimizer
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(self.loss)
                
            ############## INTERVENTION ################
            with tf.variable_scope('intervene_hidden_01'):
                self.hidden_input_1 = tf.placeholder(shape=[None, self.hidden_layer_nodes['h1']], 
                                         name='Hidden_1_placeholder', dtype=tf.float32)
                self.hidden_output_2_1 = tf.nn.relu(tf.matmul(self.hidden_input_1, self.weights['w2']) + self.biases['b2'],
                                             name = 'Hidden_2')
                self.hidden_output_3_1 = tf.nn.relu(tf.matmul(self.hidden_output_2_1, self.weights['w3']) + self.biases['b3'],
                                             name = 'Hidden_3')
                self.raw_output_1 = tf.matmul(self.hidden_output_3_1, self.weights['w4']) + self.biases['b4']
                self.final_output_1 = tf.nn.softmax(self.raw_output_1,
                                             name = 'Output_predict')

            with tf.variable_scope('intervene_hidden_02'):
                self.hidden_input_2 = tf.placeholder(shape=[None, self.hidden_layer_nodes['h2']], 
                                         name='Hidden_2_placeholder', dtype=tf.float32)
                self.hidden_output_3_2 = tf.nn.relu(tf.matmul(self.hidden_input_2, self.weights['w3']) + self.biases['b3'],
                                             name = 'Hidden_3')
                self.raw_output_2 = tf.matmul(self.hidden_output_3_2, self.weights['w4']) + self.biases['b4']
                self.final_output_2 = tf.nn.softmax(self.raw_output_2,
                                             name = 'Output_predict')

            with tf.variable_scope('intervene_hidden_03'):
                self.hidden_input_3 = tf.placeholder(shape=[None, self.hidden_layer_nodes['h3']], 
                                         name='Hidden_3_placeholder', dtype=tf.float32)
                self.raw_output_3 = tf.matmul(self.hidden_input_3, self.weights['w4']) + self.biases['b4']
                self.final_output_3 = tf.nn.softmax(self.raw_output_3,
                                             name = 'Output_predict')

            # Add information to Tensorboard            
            self.merged = tf.summary.merge_all()
                
    def train(self, X_train, y_train, print_steps, epoch):
        with tf.Session(graph = self.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.CHECK_POINT)
            train_writer = tf.summary.FileWriter(self.LOG_DIR + '/train',
                                         graph=tf.get_default_graph())
            
            # Training
            print('Training the model...')
            for i in range(1, (epoch + 1)):
                summary, _ = sess.run([self.merged, self.optimizer], feed_dict={self.X_data: X_train, self.y_target: y_train})
                train_writer.add_summary(summary, i)
                if i % print_steps == 0:
                    print('Epoch', i, '|', 'Loss:', sess.run(self.loss, feed_dict={self.X_data: X_train, self.y_target: y_train}))
            
            weight_list = [self.weights['w1'], self.weights['w2'], self.weights['w3'], self.weights['w4']]
            bias_list = [self.biases['b1'], self.biases['b2'], self.biases['b3'], self.biases['b4']] 
            saver = tf.train.Saver(weight_list + bias_list)
            saver.save(sess, os.path.join(os.getcwd(), self.CHECK_POINT))
            
    def test(self, X_test, y_test):
        with tf.Session(graph = self.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.CHECK_POINT)
            test_writer = tf.summary.FileWriter(self.LOG_DIR + '/test',
                                        graph=tf.get_default_graph())
            summary, loss = sess.run([self.merged, self.loss], feed_dict={self.X_data: X_test, self.y_target: y_test})
            print ('Testing the model...')
            print ('Loss:', loss)
            test_writer.add_summary(summary)
            
    '''
    Retrieve value of every layer (hiddens + output) in the graph
    '''            
    def predict(self, X):
        with tf.Session(graph = self.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.CHECK_POINT)
            hl_1, hl_2, hl_3, prediction = sess.run([self.hidden_output_1, self.hidden_output_2,
                                                     self.hidden_output_3, self.final_output], feed_dict={self.X_data: X})
        return hl_1, hl_2, hl_3, prediction
    
    def predict_input(self, X, sess):
        prediction = sess.run(self.final_output, feed_dict={self.X_data: X})
        return prediction
    
    def predict_hl_1(self, hl_1, sess):
        prediction = sess.run(self.final_output_1, feed_dict={self.hidden_input_1:hl_1})
        return prediction

    def predict_hl_2(self, hl_2, sess):
        prediction = sess.run(self.final_output_2, feed_dict={self.hidden_input_2:hl_2})
        return prediction

    def predict_hl_3(self, hl_3, sess):
        prediction = sess.run(self.final_output_3, feed_dict={self.hidden_input_3:hl_3})
        return prediction
        
    def extract_weights_biases(self, X_train):
        with tf.Session(graph = self.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.CHECK_POINT)
            weights, biases = sess.run([self.weights, self.biases], feed_dict={self.X_data: X_train})
            
        return weights, biases
        
def proper_use():
    def read_dataset():
        dataset = IRISDataset()
        X_test, X_train, y_test, y_train = dataset.split_to_train_test(seed=10)
        return X_test, X_train, y_test, y_train
        
    def train_model(X_test, X_train, y_test, y_train):
        model = NeuralNetwork()
        model.create_graph()
        return model        
        
    def use_model(X_train):
        model = NeuralNetwork()
        model.create_graph()
        
        # get input data for each layer
        hl_1, hl_2, hl_3, prediction = model.predict(X_train)
        
        # to save time reload weight, using session wrapped outside could help
        # decreasing running time, A LOT!
        with tf.Session(graph = model.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, model.CHECK_POINT)
            prediction = model.predict_input(X_train, sess)
            prediction = model.predict_hl_1(hl_1, sess)
            prediction = model.predict_hl_2(hl_2, sess)
            prediction = model.predict_hl_3(hl_3, sess)
        
    X_test, X_train, y_test, y_train = read_dataset()
    model = train_model(X_test, X_train, y_test, y_train)
    use_model(X_train)
    weights, biases = model.extract_weights_biases(X_train)
    
if __name__ == '__main__':
    proper_use()