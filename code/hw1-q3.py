#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils

def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        y_hat = self.predict(x_i)

        if y_i != y_hat:
            self.W[y_i] += x_i
            self.W[y_hat] -= x_i



class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        y_hat = self.predict(x_i)

        if y_i != y_hat:
            self.W[y_i] += learning_rate*x_i

            z_x = np.sum(np.exp(np.dot(self.W, x_i.T)))
            
            for i in range(0, len(self.W)):
                p_w = np.exp(np.dot(self.W[i], x_i.T))
                self.W[i] -= learning_rate*(np.dot((p_w/z_x), x_i.T))


class MLP(object):
    def __init__(self, n_classes, n_features, hidden_size, hidden_layers = 1):
        # Initialize an MLP with a single hidden layer.

        rng = np.random.default_rng()

        assert(hidden_layers >= 1)
        self.weights = [rng.normal(loc=0.1, scale=0.1, size=(hidden_size, n_features))]     # First layer
        self.weights += [rng.normal(loc=0.1, scale=0.1, size=(hidden_size, hidden_size)) for i in range(hidden_layers - 1)]
        self.weights.append(rng.normal(loc=0.1, scale=0.1, size=(n_classes, hidden_size)))  # Output layer

        self.biases = [np.zeros(hidden_size) for i in range(hidden_layers)]                 # All hidden layers
        self.biases.append(np.zeros(n_classes))                                             # Output layer

    def relu(self, X):
        return np.maximum(0, X)

    def softmax(self, X):
        z = X - X.max()     # To make softmax numerically stable
        return np.exp(z) / np.sum(np.exp(z), axis=0)

    def pre_activation(self, X, weights, bias):
        return weights @ X + bias

    def forward(self, X):
        """
        X: vector with n_features

        Returns: output, hidden_activations
                    output:             the result of the softmax from the last layer
                    hidden_activations: the activations of the hidden layers
        """

        hidden_activations = []
        n_layers = len(self.weights)
        output = None

        for i in range(n_layers):
            h = X if i == 0 else hidden_activations[i - 1]
            z = self.pre_activation(h, self.weights[i], self.biases[i])

            if i < n_layers - 1:
                hidden_activations.append(self.relu(z))
            else:
                output = self.softmax(z)

        return output, hidden_activations
    
    def predict(self, X):
        """
        X: vector with n_features

        Returns: label: a one-hot vector that represents the class that the
                        network predicts for the given vector.
        """

        output, _ = self.forward(X)
        label = np.zeros_like(output)
        label[np.argmax(output)] = 1

        return label

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """

        preds = []
        for ix in range(X.shape[0]):
            y_hat = self.predict(X[ix])
            preds.append(np.argmax(y_hat))

        y_hat = np.array(preds)
        n_correct = np.equal(y_hat, y).sum()
        n_possible = y.shape[0]

        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        def relu_derivative(X):
            return (X > 0) * 1
        
        def update_weights_and_biases(grad_weights, grad_biases):
            for i in range(n_layers):
                self.weights[i] -= learning_rate * grad_weights[i]
                self.biases[i]  -= learning_rate * grad_biases[i]

        n_layers = len(self.weights)

        for i in range(X.shape[0]):
            grad_weights    = []
            grad_biases     = []

            output, hidden_activations = self.forward(X[i])

            # softmax_c(z(x)) - 1(y = c)
            true_label = np.zeros_like(output)
            true_label[y[i]] = 1
            grad_z = output - true_label

            for j in range(n_layers - 1, -1, -1):
                h = X[i] if j == 0 else hidden_activations[j - 1]

                grad_w = grad_z[:, None] @ h[:, None].T
                grad_weights.append(grad_w)

                grad_biases.append(grad_z)

                grad_h = self.weights[j].T @ grad_z

                grad_z = grad_h * relu_derivative(h)

            grad_weights.reverse()
            grad_biases.reverse()

            update_weights_and_biases(grad_weights, grad_biases)

def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.savefig("hw1-q3-mlp.png", dpi=300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size, opt.layers)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()
