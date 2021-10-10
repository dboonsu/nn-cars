#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################


import numpy as np
import csv
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


class NeuralNet:
    def __init__(self, dataFile, header=True):
        self.raw_input = pd.read_csv(dataFile)




    # TODO: Write code for pre-processing the dataset, which would include
    # standardization, normalization,
    #   categorical to numerical, etc
    def preprocess(self):
        # Replaces all categorial data into numeral data
        self.rawinput = self.raw_input.replace(('vhigh', 'high', 'med', 'low', 'big', 'small','5more','more'), (4,3,2,1,3,1,5,6), inplace=True)
        self.processed_data = self.raw_input
        return 0

    # TODO: Train and evaluate models for all combinations of parameters
    # specified in the init method. We would like to obtain following outputs:
    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot should be color coded i.e.
    #       different color for each model

    def train_evaluate(self):
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols-1)]
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        # Below are the hyperparameters that you need to use for model
        #   evaluation
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [.01, .1]
        learning_rate_selected = learning_rate[1]
        max_iterations = [100, 200] # also known as epochs
        max_iterations_selected = max_iterations[1]
        num_hidden_layers = [2, 3]
        num_hidden_layers_selected = num_hidden_layers[1]
        classes = ['vgood', 'good', 'acc', 'unacc']
        temp = "8"
        # Create the neural network and be sure to keep track of the performance
        #   metrics
        NNLog = MLPClassifier(activation=activations[0], learning_rate_init=learning_rate_selected, max_iter=max_iterations_selected, hidden_layer_sizes=(num_hidden_layers_selected,))
        NNTanh = MLPClassifier(activation=activations[1], learning_rate_init=learning_rate_selected, max_iter=max_iterations_selected, hidden_layer_sizes=(num_hidden_layers_selected,))
        NNRelu = MLPClassifier(activation=activations[1], learning_rate_init=learning_rate_selected, max_iter=max_iterations_selected, hidden_layer_sizes=(num_hidden_layers_selected,))

        #NN.fit(X_train, y_train)
        with open('output.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Learning Rate: " + str(learning_rate_selected),
                             " Max Iterations: " + str(max_iterations_selected),
                             " Num Hidden Layers: " + str(num_hidden_layers_selected)])
            writer.writerow(["Iteration",
                             "Set " + temp + " Testing Logistic Accuracy",
                             "Set " + temp + " Testing TanH Accuracy",
                             "Set " + temp + " Testing ReLu Accuracy",
                             "Logistic Loss",
                             "TanH Loss",
                             "ReLu Loss",
                             "Set " + temp + " Training Logistic Accuracy",
                             "Set " + temp + " Training TanH Accuracy",
                             "set " + temp + " Training ReLu Accuracy"])
            for iter in range(max_iterations[0]):
                NNLog.partial_fit(X_train, y_train, classes)
                NNTanh.partial_fit(X_train, y_train, classes)
                NNRelu.partial_fit(X_train, y_train, classes)
                writer.writerow([str(iter),
                                 NNLog.score(X_test, y_test),
                                 NNTanh.score(X_test, y_test),
                                 NNRelu.score(X_test, y_test),
                                 NNLog.loss_,
                                 NNTanh.loss_,
                                 NNRelu.loss_,
                                 NNLog.score(X_train, y_train),
                                 NNTanh.score(X_train, y_train),
                                 NNRelu.score(X_train, y_train)])

            # Plot the model history for each model in a single plot
            # model history is a plot of accuracy vs number of epochs
            # you may want to create a large sized plot to show multiple lines
            # in a same figure.

        return 0




if __name__ == "__main__":
    neural_network = NeuralNet("https://raw.githubusercontent.com/dboonsu/nn-cars/main/car.csv") # put in path to your file
    neural_network.preprocess()
    neural_network.train_evaluate()
