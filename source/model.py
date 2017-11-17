import tensorflow as tf
import os
import numpy as np
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
    
    def __init__(self):

        self.graph = tf.Graph()
        with self.graph.as_default():

            input_layer_nodes = 4; output_layer_nodes = 3            
            hidden_layer_nodes = {'h1': 8, 'h2': 6, 'h3': 4}

            # Initialize placeholders
            self.X_data = tf.placeholder(shape=[None, input_layer_nodes], 
                                         name='Input', dtype=tf.float32)
            self.y_target = tf.placeholder(shape=[None, output_layer_nodes],
                                         name='Labels', dtype=tf.float32)
            
            
            # Create variables for Neural Network layers
            self.weights = {
            'w1': tf.Variable(tf.random_normal(shape=[input_layer_nodes,
                                                      hidden_layer_nodes['h1']]),
                                                        name = 'W_1'),
            'w2': tf.Variable(tf.random_normal(shape=[hidden_layer_nodes['h1'],
                                                      hidden_layer_nodes['h2']]),
                                                        name = 'W2'),
            'w3': tf.Variable(tf.random_normal(shape=[hidden_layer_nodes['h2'],
                                                      hidden_layer_nodes['h3']]),
                                                        name = 'W3'),
            'w4': tf.Variable(tf.random_normal(shape=[hidden_layer_nodes['h3'],
                                                      output_layer_nodes]),
                                                        name = 'W4')
            }
            variable_summaries(self.weights['w1'], 'Weight_Input_Hidden1')
            variable_summaries(self.weights['w2'], 'Weight_Hidden1_Hidden2')
            variable_summaries(self.weights['w3'], 'Weight_Hidden2_Hidden3')
            variable_summaries(self.weights['w4'], 'Weight_Hidden3_Output')
            
            self.biases = {
            'b1': tf.Variable(tf.random_normal(shape=[hidden_layer_nodes['h1']]),
                              name = 'b1'),
            'b2': tf.Variable(tf.random_normal(shape=[hidden_layer_nodes['h2']]),
                              name = 'b2'),
            'b3': tf.Variable(tf.random_normal(shape=[hidden_layer_nodes['h3']]),
                              name = 'b3'),
            'b4': tf.Variable(tf.random_normal(shape=[output_layer_nodes]),
                              name = 'b4')
            }
            variable_summaries(self.biases['b1'], 'Biases_Input_Hidden1')
            variable_summaries(self.biases['b2'], 'Biases_Hidden1_Hidden2')
            variable_summaries(self.biases['b3'], 'Biases_Hidden2_Hidden3')
            variable_summaries(self.biases['b4'], 'Biases_Hidden3_Hidden4')

    
            # Operations
            self.hidden_output_1 = tf.nn.relu(tf.matmul(self.X_data, self.weights['w1']) + self.biases['b1'],
                                         name = 'Hidden_1')
            self.hidden_output_2 = tf.nn.relu(tf.matmul(self.hidden_output_1, self.weights['w2']) + self.biases['b2'],
                                         name = 'Hidden_2')
            self.hidden_output_3 = tf.nn.relu(tf.matmul(self.hidden_output_2, self.weights['w3']) + self.biases['b3'],
                                         name = 'Hidden_3')
            self.final_output = tf.matmul(self.hidden_output_3, self.weights['w4']) + self.biases['b4']
                                         
            # Cost Function
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.final_output, labels=self.y_target))
            tf.summary.scalar('loss', self.loss)
            
            # Optimizer
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(self.loss)
            
            self.merged = tf.summary.merge_all()

            ########## MODIFY FOR SHAPLEY ###########
            self.hidden_output_1_hd = tf.placeholder(shape=[None, hidden_layer_nodes['h1']], 
                                         name='Hidden_1_placeholder', dtype=tf.float32)
            hidden_output_1_2 = tf.nn.relu(tf.matmul(self.hidden_output_1_hd, self.weights['w2']) + self.biases['b2'])
            hidden_output_1_3 = tf.nn.relu(tf.matmul(hidden_output_1_2, self.weights['w3']) + self.biases['b3'])
            self.final_output_1 = tf.matmul(hidden_output_1_3, self.weights['w4']) + self.biases['b4'] 
            
                                               
            self.hidden_output_2_hd = tf.placeholder(shape=[None, hidden_layer_nodes['h2']], 
                                         name='Hidden_2_placeholder', dtype=tf.float32)
            hidden_output_2_3 = tf.nn.relu(tf.matmul(self.hidden_output_2_hd, self.weights['w3']) + self.biases['b3'])
            self.final_output_2 = tf.matmul(hidden_output_2_3, self.weights['w4']) + self.biases['b4']
                  
                                               
            self.hidden_output_3_hd = tf.placeholder(shape=[None, hidden_layer_nodes['h3']], 
                                         name='Hidden_3_placeholder', dtype=tf.float32)
            self.final_output_3 = tf.matmul(self.hidden_output_3_hd, self.weights['w4']) + self.biases['b4']

    def train(self, X_train, y_train, print_steps, epoch):
        with tf.Session(graph = self.graph) as sess:
            self.saver = tf.train.Saver()
            train_writer = tf.summary.FileWriter(self.LOG_DIR + '/train',
                                         graph=tf.get_default_graph())

            # Initialize variables
            sess.run(tf.global_variables_initializer())
            
            # Training
            print('Training the model...')
            for i in range(1, (epoch + 1)):
                summary, _ = sess.run([self.merged, self.optimizer], feed_dict={self.X_data: X_train, self.y_target: y_train})
                train_writer.add_summary(summary, i)
                if i % print_steps == 0:
                    print('Epoch', i, '|', 'Loss:', sess.run(self.loss, feed_dict={self.X_data: X_train, self.y_target: y_train}))
            
            self.saver.save(sess, os.path.join(os.getcwd(), 'saved_model_01'))
            
    def test(self, X_test, y_test):
        with tf.Session(graph = self.graph) as sess:
            new_saver = tf.train.import_meta_graph(os.path.join(os.getcwd(),
                                                                'saved_model_01.meta'))
            new_saver.restore(sess, tf.train.latest_checkpoint('./'))
            test_writer = tf.summary.FileWriter(self.LOG_DIR + '/test',
                                        graph=tf.get_default_graph())
            summary, loss = sess.run([self.merged, self.loss], feed_dict={self.X_data: X_test, self.y_target: y_test})
            print ('Testing the model...')
            print ('Loss:', loss)
            test_writer.add_summary(summary)
            
    def predict(self, X):
        with tf.Session(graph = self.graph) as sess:
            new_saver = tf.train.import_meta_graph(os.path.join(os.getcwd(),
                                                                'saved_model_01.meta'))
            new_saver.restore(sess, tf.train.latest_checkpoint('./'))
            hl_1, hl_2, hl_3, prediction = sess.run([self.hidden_output_1, self.hidden_output_2,
                                                     self.hidden_output_3, self.final_output], feed_dict={self.X_data: X})
        return hl_1, hl_2, hl_3, prediction
    
    def predict_input(self, X, sess):
        prediction = sess.run(self.final_output, feed_dict={self.X_data: X})
        return prediction
    
    def predict_hl_1(self, hl_1, sess):
        prediction = sess.run(self.final_output_1, feed_dict={self.hidden_output_1_hd:hl_1})
        return prediction

    def predict_hl_2(self, hl_2, sess):
        prediction = sess.run(self.final_output_2, feed_dict={self.hidden_output_2_hd:hl_2})
        return prediction

    def predict_hl_3(self, hl_3, sess):
        prediction = sess.run(self.final_output_3, feed_dict={self.hidden_output_3_hd:hl_3})
        return prediction

        
def test_case_1():
    print_steps = 50; epoch = 100
    dataset = IRISDataset()
    X_test, X_train, y_test, y_train = dataset.split_to_train_test(seed=10)
    
    model = NeuralNetwork()
    model.train(X_train, y_train, print_steps, epoch)
    model.test(X_test, y_test)
    
    hl_1, hl_2, hl_3, prediction = model.predict(X_train)
    print (hl_1.shape, hl_2.shape, hl_3.shape)
    print (prediction.shape)
    
    with tf.Session(graph = model.graph) as sess:
        new_saver = tf.train.import_meta_graph(os.path.join(os.getcwd(),
                                                                'saved_model_01.meta'))
        new_saver.restore(sess, tf.train.latest_checkpoint('./'))
        prediction = model.predict_input(X_train, sess)
        print (prediction.shape)
        prediction = model.predict_hl_1(hl_1, sess)
        print (prediction.shape)
        prediction = model.predict_hl_2(hl_2, sess)
        print (prediction.shape)
        prediction = model.predict_hl_3(hl_3, sess)
        print (prediction.shape)
        
if __name__ == '__main__':
    test_case_1()        

    