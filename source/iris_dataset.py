import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
from mpl_toolkits.mplot3d import Axes3D    

class IRISDataset(object):
    def __init__(self):
        # Load the dataset
        self._dataset = pd.read_csv('Iris_Dataset.csv')
      
        # Split Species column into 3 columns corresponding to 3 flower types
        dataset = pd.get_dummies(self._dataset, columns=['Species'])
        values = list(dataset.columns.values)

        # Convert labels to 1 hot encoding vector
        y = dataset[values[-3:]]
        self._y = np.array(y, dtype='float32')
        X = dataset[values[1:-3]]
        self._X = np.array(X, dtype='float32')    
        
    def split_to_train_test(self, seed = None):
        
        if (seed != None):
            np.random.seed(seed)
        # Shuffle Data
        indices = np.random.choice(len(self._X), len(self._X), replace=False)
        X_values = self._X[indices]
        y_values = self._y[indices]
        
        # Creating a Train and a Test Dataset
        test_size = 10
        X_test = X_values[-test_size:]
        X_train = X_values[:-test_size]
        y_test = y_values[-test_size:]
        y_train = y_values[:-test_size]
        
        return X_test, X_train, y_test, y_train
        
    def plot_2D(self):
        sns.set(style="white", color_codes=True)    
        sns.pairplot(self._dataset.drop("Id", axis=1), hue="Species", size=3)

    def plot_3D(self):
        dataset = self._dataset
        species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        features = list(dataset.columns.values)[1:-1]
        ax = [None] * 4
        c = ['b', 'g', 'r']
        
        fig = plt.figure(figsize=(16, 16))

        for drop_feature in range(4):
            ax[drop_feature] = fig.add_subplot(2, 2, drop_feature+1, projection='3d')
    
            for i in range(3):
                chosen_species = species[i]
                index = list(dataset['Species'] == chosen_species)
                chosen_features = np.array(dataset[features[:drop_feature] + features[drop_feature+1:]])
                chosen_features = chosen_features[index]
                x = chosen_features[:,0]; y = chosen_features[:,1]; z = chosen_features[:,2]
                ax[drop_feature].scatter(x, y, z, c=c[i], marker='o')
                
                x_label, y_label, z_label = features[:drop_feature] + features[drop_feature+1:]
                ax[drop_feature].set_xlabel(x_label);    
                ax[drop_feature].set_ylabel(y_label);
                ax[drop_feature].set_zlabel(z_label);

        plt.show()
    
def test_case_1():
    dataset = IRISDataset()
    dataset.plot_2D()
    dataset.plot_3D()
    X_test, X_train, y_test, y_train = dataset.split_to_train_test(seed=30)
    print (y_train)
    print (dataset._dataset.columns.values)
    
if __name__ == '__main__':
    test_case_1()