#!/usr/bin/env python

# Deep Learning Homework 2

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
import numpy as np

import utils

class CNN(nn.Module):
    
    def __init__(self, dropout_prob, no_maxpool=False):
        super(CNN, self).__init__()
        self.no_maxpool = no_maxpool
        if not no_maxpool:
            # Create the convulutional layers
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding='same')
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0)

        else:
            # Create the convulutional layers
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0)

        # Create the ReLU activation function
        self.relu = nn.ReLU()

        # Create the max pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Create the dropout layer
        self.drop = nn.Dropout(p=dropout_prob)

        # Create the affine transformation layers
        self.fc1 = nn.Linear(16 * 6 * 6, 320)
        self.fc2 = nn.Linear(320, 120)
        self.fc3 = nn.Linear(120, 4)

    def forward(self, x):
        # input should be of shape [b, c, w, h]
        # conv and relu layers
        x = self.conv1(x)
        x = self.relu(x)

        # max-pool layer if using it
        if not self.no_maxpool:
            x = self.maxpool(x)
        
        # conv and relu layers
        x = self.conv2(x)
        x = self.relu(x)
        
        # max-pool layer if using it
        if not self.no_maxpool:
            x = self.maxpool(x)
        
        # prep for fully connected layer + relu
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        
        # drop out
        x = self.drop(x)

        # second fully connected layer + relu
        x = self.fc2(x)
        x = self.relu(x)
        
        # last fully connected layer
        x = self.fc3(x)
        
        return F.log_softmax(x,dim=1)

def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function
    """
    optimizer.zero_grad()
    out = model(X, **kwargs)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, X):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)
    return predicted_labels


def evaluate(model, X, y):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    y_hat = predict(model, X)
    n_correct = (y == y_hat).sum().item()
    n_possible = float(y.shape[0])
    model.train()
    return n_correct / n_possible


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('plots/%s.png' % (name), bbox_inches='tight')


def get_number_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=8, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01,
                        help="""Learning rate for parameter updates""")
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-dropout', type=float, default=0.7)
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-no_maxpool', action='store_true')
    
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    # Set up the device for computation
    if torch.cuda.is_available():
        device = torch.device('cuda')
    # elif torch.backends.mps.is_available():
    #     device = torch.device('mps')
    else:
        device = torch.device('cpu')

    data = utils.load_oct_data()
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True)
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y
    dev_X = torch.reshape(dev_X, (dev_X.size()[0], 1, 28, 28))
    test_X = torch.reshape(test_X, (test_X.size()[0], 1, 28, 28))
    dev_X, dev_y = dev_X.to(device), dev_y.to(device)
    test_X, test_y = test_X.to(device), test_y.to(device)

    # initialize the model
    model = CNN(opt.dropout, no_maxpool=opt.no_maxpool).to(device)
    
    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay
    )
    
    # get a loss criterion
    criterion = nn.NLLLoss()
    
    # training loop
    epochs = np.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        for X_batch, y_batch in train_dataloader:
            X_batch = torch.reshape(X_batch, (X_batch.size()[0], 1, 28, 28))
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        valid_accs.append(evaluate(model, dev_X, dev_y))
        print('Valid acc: %.4f' % (valid_accs[-1]))

    print('Final Test acc: %.4f' % (evaluate(model, test_X, test_y)))
    # plot
    config = "{}-{}-{}-{}-{}".format(opt.learning_rate, opt.dropout, opt.l2_decay, opt.optimizer, opt.no_maxpool)

    plot(epochs, train_mean_losses, ylabel='Loss', name='CNN-training-loss-{}'.format(config))
    plot(epochs, valid_accs, ylabel='Accuracy', name='CNN-validation-accuracy-{}'.format(config))
    
    model.train()
    print('Number of trainable parameters: ', get_number_trainable_params(model))

if __name__ == '__main__':
    main()
