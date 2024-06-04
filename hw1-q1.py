#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils


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

        # Compute the prediction
        y_hat = self.predict(x_i)

        # Update the weights if the prediction is wrong
        if y_hat != y_i:
            self.W[y_i, :] += x_i
            self.W[y_hat, :] -= x_i

        return 


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """

        # Compute the scores for each class
        scores = np.dot(self.W, x_i)

        # Compute the probality distribution of the scores using softmax
        prob_dist = np.exp(scores) / np.sum(np.exp(scores))

        # Compute the one hot encoding of the gold label
        y_one_hot = np.zeros(self.W.shape[0])
        y_one_hot[y_i] = 1

        # Compute the gradient of the loss with respect to the scores
        grad = prob_dist - y_one_hot

        # Update the weights
        self.W -= learning_rate * np.outer(grad, x_i)

        return


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().

    # Initialize an MLP with a single hidden layer.
    def __init__(self, n_classes, n_features, hidden_size):

        # Initialize the weights of the hidden layer with a Gaussian distribution of mean 0.1 and standard deviation 0.1
        self.W_h = np.random.normal(0.1, 0.1, (hidden_size, n_features))

        # Initialize the biases of the hidden layer with zeros
        self.b_h = np.zeros(hidden_size)

        # Initialize the weights of the output layer with a Gaussian distribution of mean 0.1 and standard deviation 0.1
        self.W_o = np.random.normal(0.1, 0.1, (n_classes, hidden_size))

        # Initialize the biases of the output layer zeros
        self.b_o = np.zeros(n_classes)

        return

    # Compute the forward pass of the network given the input data X.
    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.

        # Compute the scores of the hidden layer for all examples
        hidden_scores = np.dot(X, self.W_h.T) + self.b_h

        # Compute the activation of the hidden layer using ReLU
        hidden_activation = np.maximum(0, hidden_scores)

        # Compute the scores of the output layer for all examples
        output_scores = np.dot(hidden_activation, self.W_o.T) + self.b_o

        # Compute the probability distribution of the scores using softmax
        prob_dist = np.exp(output_scores - np.max(output_scores, axis=1, keepdims=True)) / np.sum(np.exp(output_scores - np.max(output_scores, axis=1, keepdims=True)), axis=1, keepdims=True)

        # Compute the predicted labels for all examples
        pred_labels = np.argmax(prob_dist, axis=1)

        return pred_labels
    
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
        """
        Dont forget to return the loss of the epoch.
        """

        # Initialize the loss of the epoch
        loss = 0

        for x_i, y_i in zip(X, y):
            ## Forward pass

            # Compute the scores of the hidden layer
            hidden_scores = np.dot(self.W_h, x_i) + self.b_h

            # Compute the activation of the hidden layer using ReLU
            hidden_activation = np.maximum(0, hidden_scores)

            # Compute the scores of the output layer
            output_scores = np.dot(self.W_o, hidden_activation) + self.b_o

            # Compute the probability distribution of the scores using softmax (subtracting the maximum to avoid numerical instability)
            prob_dist = np.exp(output_scores - np.max(output_scores)) / np.sum(np.exp(output_scores - np.max(output_scores)))

            # Compute the cross entropy loss of the epoch
            loss += -np.log(prob_dist[y_i])

            # Compute the one hot encoding of the gold label
            y_one_hot = np.zeros(self.W_o.shape[0])
            y_one_hot[y_i] = 1

            ## Backward pass

            # Compute the gradient of the loss with respect to the scores
            grad = prob_dist - y_one_hot

            # Compute the gradient of the loss with respect to the output layer weights
            grad_W_o = np.outer(grad, hidden_activation)

            # Compute the gradient of the loss with respect to the output layer biases
            grad_b_o = grad

            # Compute the gradient of the loss with respect to the hidden layer activations
            grad_h_h = np.dot(self.W_o.T, grad)

            # Compute the gradient of the loss with respect to the hidden layer scores
            grad_z_h = grad_h_h * (hidden_activation > 0)

            # Compute the gradient of the loss with respect to the hidden layer weights
            grad_W_h = np.outer(grad_z_h, x_i)

            # Compute the gradient of the loss with respect to the hidden layer biases
            grad_b_h = grad_z_h

            # Update the weights and biases
            self.W_o -= learning_rate * grad_W_o
            self.b_o -= learning_rate * grad_b_o
            self.W_h -= learning_rate * grad_W_h
            self.b_h -= learning_rate * grad_b_h
       
        return loss

def plot(epochs, train_accs, val_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    plt.show()

def plot_loss(epochs, loss):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
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
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_oct_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []

    # Print sizes of the datasets
    print('Training set size: {}'.format(train_X.shape[0]))
    print('Validation set size: {}'.format(dev_X.shape[0]))
    print('Test set size: {}'.format(test_X.shape[0]))
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs)
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss)


if __name__ == '__main__':
    main()
