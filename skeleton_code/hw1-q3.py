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
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.

        rng = np.random.default_rng()

        self.l1_weights = rng.normal(loc=0.1, scale=0.1 ** 2, size=(hidden_size, n_features))
        self.out_weights = rng.normal(loc=0.1, scale=0.1 ** 2, size=(n_classes, hidden_size))

        self.l1_bias = np.zeros(hidden_size)
        self.out_bias = np.zeros(n_classes)

    def pre_activation(self, X, weights, bias):
        return weights @ X + bias

    def l1_pre_activation(self, X):
        return self.pre_activation(X.T, self.l1_weights, self.l1_bias)

    def out_pre_activation(self, X):
        return self.pre_activation(X, self.out_weights, self.out_bias)

    def l1_activation(self, l1_pre_activation):
        return np.maximum(0, l1_pre_activation)

    def out_activation(self, out_preactivation):
        def softmax(X):
            
            print(X)

            num = np.exp(X)
            den = np.sum(np.exp(X))

            print(num, den)

            probs = num / den

            return probs
        return softmax(out_preactivation)

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.

        l1_activation = self.l1_activation(self.l1_pre_activation(X))

        return self.out_activation(self.out_pre_activation(l1_activation))

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        def update_weights_and_biases(grad_w1, grad_w2, grad_b1, grad_b2):
            self.l1_weights     -= learning_rate * grad_w1
            self.l1_bias        -= learning_rate * grad_b1
            self.out_weights    -= learning_rate * grad_w2
            self.out_bias       -= learning_rate * grad_b2

        for i in range(len(X)):
            pred = self.predict(X[i])
            h1 = self.l1_activation(self.l1_pre_activation(X[i]))


            grad_z2 = pred - y[i]
            print(grad_z2, len(grad_z2))
            print(h1.T, len(h1.T))
            grad_weights_2 = grad_z2 @ h1.T
            grad_biases_2 = grad_z2

            grad_h1 = self.out_weights.T @ grad_z2

            grad_z1 = grad_h1 * (1 - h1 ** 2)

            grad_weights_1 = grad_z1 @ X[i].T
            grad_biases_1 = grad_z1

            update_weights_and_biases(
                                        grad_weights_1,
                                        grad_weights_2,
                                        grad_biases_1,
                                        grad_biases_2)

def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


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
        model = MLP(n_classes, n_feats, opt.hidden_size)
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
