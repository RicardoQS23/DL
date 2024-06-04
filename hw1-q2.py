#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt

import utils

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

# Q2.1
class LogisticRegression(nn.Module):

    def __init__(self, n_classes, n_features, **kwargs):
        """
        n_classes (int)
        n_features (int)

        The __init__ should be used to declare what kind of layers and other
        parameters the module has. For example, a logistic regression module
        has a weight matrix and bias vector. For an idea of how to use
        pytorch to make weights and biases, have a look at
        https://pytorch.org/docs/stable/nn.html
        """
        super().__init__()
        # In a pytorch module, the declarations of layers needs to come after
        # self.W = nn.Parameter(torch.zeros(n_classes, n_features))
        # self.b = nn.Parameter(torch.zeros(n_classes))
        self.Layer = nn.Linear(n_features, n_classes)
        # nn.Parameter is a special kind of Tensor, that will get automatically
        # registered as Module's parameter once it's assigned as an attribute.
        # nn.Parameters require gradients by default.

        # Note that you don't need to declare the variables you use in this
        # module. For example, to implement a module that computes
        # y = Wx + b
        # you don't need to declare y, W or x. You only need to define the
        # parameters you need (in this case, W and b), and the forward()
        # function that computes y from W and b and x.


        # the super __init__ line, otherwise the magic doesn't work.
        # You can also do other fun stuff here like initializing weights
        # and biases, though we don't require you to.


    def forward(self, x, **kwargs):
        """
        x (batch_size x n_features): a batch of training examples

        Every subclass of nn.Module needs to have a forward() method. forward()
        describes how the module computes the forward pass. In a log-lineear
        model like this, for example, forward() needs to compute the logits
        y = Wx + b, and return y (you don't need to worry about taking the
        softmax of y because nn.CrossEntropyLoss does that for you).

        One nice thing about pytorch is that you only need to define the
        forward pass -- this is enough for it to figure out how to do the
        backward pass.

        The forward() function needs to return the logits (y) as output,
        but the caller only cares about the loss, so you can ignore the
        return value of this function.
        """

        # y = torch.matmul(x, self.W.t()) + self.b
        y = self.Layer(x)

        return y


# Q2.2
class FeedforwardNetwork(nn.Module):
    def __init__(
            self, n_classes, n_features , hidden_size = 200, layers = 2,
            activation_type = "relu", dropout = 0.0, **kwargs):
        """
        n_classes (int)
        n_features (int)
        hidden_size (int)
        layers (int)
        activation_type (str)
        dropout (float): dropout probability

        As in logistic regression, the __init__ here defines a bunch of
        attributes that each FeedforwardNetwork instance has. Note that nn
        includes modules for several activation functions and dropout as well.
        """
        super().__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.activation_type = activation_type
        self.dropout = dropout

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(n_features, hidden_size))
        for _ in range(layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.hidden_layers.append(nn.Linear(hidden_size, n_classes))

        self.activation = nn.ModuleDict({
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
        })

        self.dropout_layer = nn.Dropout(p=dropout)


    def forward(self, x, **kwargs):
        """
        x (batch_size x n_features): a batch of training examples

        This method needs to perform all the computation needed to compute
        the output logits from x. This will include using various hidden
        layers, pointwise nonlinear functions, and dropout.
        """

        h = x

        for i in range(self.layers):
            z = self.hidden_layers[i](h)
            h = self.activation[self.activation_type](z)
            h = self.dropout_layer(h)

        y = self.hidden_layers[-1](h)

        return y


def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function

    To train a batch, the model needs to predict outputs for X, compute the
    loss between these predictions and the "gold" labels y using the criterion,
    and compute the gradient of the loss with respect to the model parameters.

    Check out https://pytorch.org/docs/stable/optim.html for examples of how
    to use an optimizer object to update the parameters.

    This function should return the loss (tip: call loss.item()) to get the
    loss as a numerical value that is not part of the computation graph.
    """
    model.train()
    optimizer.zero_grad()
    logits = model(X.to(DEVICE))
    loss = criterion(logits, y.to(DEVICE))
    loss.backward()
    optimizer.step()

    # loss = 0

    # for x_i, y_i in zip(X, y):
    #     model.train()
    #     optimizer.zero_grad()
    #     logits = model(x_i.unsqueeze(0))
    #     loss += criterion(logits, y_i.unsqueeze(0))
    #     loss.backward()
    #     optimizer.step()

    return loss.item()


def predict(model, X):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)
    return predicted_labels


@torch.no_grad()
def evaluate(model, X, y, criterion):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    y = y.to(DEVICE)
    model.eval()
    logits = model(X.to(DEVICE))
    loss = criterion(logits, y)
    loss = loss.item()
    y_hat = logits.argmax(dim=-1)
    n_correct = (y == y_hat).sum().item()
    n_possible = float(y.shape[0])
    model.train()
    return loss, n_correct / n_possible


def plot(epochs, plottables, name='', ylim=None):
    """Plot the plottables over the epochs.
    
    Plottables is a dictionary mapping labels to lists of values.
    """
    plt.clf()
    plt.xlabel('Epoch')
    for label, plottable in plottables.items():
        plt.plot(epochs, plottable, label=label)
    plt.legend()
    if ylim:
        plt.ylim(ylim)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=1, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01)
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-hidden_size', type=int, default=100)
    parser.add_argument('-layers', type=int, default=1)
    parser.add_argument('-dropout', type=float, default=0.3)
    parser.add_argument('-activation',
                        choices=['tanh', 'relu'], default='relu')
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    data = utils.load_oct_data()
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))

    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    n_classes = torch.unique(dataset.y).shape[0]  # 10
    n_feats = dataset.X.shape[1]

    # if torch.cuda.is_available():
    #     device = torch.device('cuda')
    # elif torch.backends.mps.is_available():
    #     device = torch.device('mps')
    # else:
    #     device = torch.device('cpu')

    # dev_X, dev_y = dev_X.to(device), dev_y.to(device)
    # test_X, test_y = test_X.to(device), test_y.to(device)

    # initialize the model
    if opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats).to(DEVICE)
    else:
        model = FeedforwardNetwork(
            n_classes,
            n_feats,
            opt.hidden_size,
            opt.layers,
            opt.activation,
            opt.dropout
        ).to(DEVICE)

    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(),
        lr=opt.learning_rate,
        weight_decay=opt.l2_decay)

    # get a loss criterion
    criterion = nn.CrossEntropyLoss()

    # training loop
    epochs = torch.arange(1, opt.epochs + 1)
    train_losses = []
    valid_losses = []
    valid_accs = []
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        epoch_train_losses = []
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            epoch_train_losses.append(loss)

        epoch_train_loss = torch.tensor(epoch_train_losses).mean().item()
        val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion)

        print('Training loss: %.4f' % epoch_train_loss)
        print('Valid acc: %.4f' % val_acc)

        train_losses.append(epoch_train_loss)
        valid_losses.append(val_loss)
        valid_accs.append(val_acc)

    _, test_acc = evaluate(model, test_X, test_y, criterion)
    print('Final Test acc: %.4f' % (test_acc))
    # plot
    if opt.model == "logistic_regression":
        config = (
            f"batch-{opt.batch_size}-lr-{opt.learning_rate}-epochs-{opt.epochs}-"
            f"l2-{opt.l2_decay}-opt-{opt.optimizer}"
        )
    else:
        config = (
            f"batch-{opt.batch_size}-lr-{opt.learning_rate}-epochs-{opt.epochs}-"
            f"hidden-{opt.hidden_size}-dropout-{opt.dropout}-l2-{opt.l2_decay}-"
            f"layers-{opt.layers}-act-{opt.activation}-opt-{opt.optimizer}"
        )

    losses = {
        "Train Loss": train_losses,
        "Valid Loss": valid_losses,
    }
    # Choose ylim based on model since logistic regression has higher loss
    if opt.model == "logistic_regression":
        ylim = (0., 1.6)
    elif opt.model == "mlp":
        ylim = (0., 1.2)
    else:
        raise ValueError(f"Unknown model {opt.model}")
    
    plot(epochs, losses, name=f'{opt.model}-training-loss-{config}', ylim=ylim)
    accuracy = { "Valid Accuracy": valid_accs }
    plot(epochs, accuracy, name=f'{opt.model}-validation-accuracy-{config}', ylim=(0., 1.))


if __name__ == '__main__':
    main()
