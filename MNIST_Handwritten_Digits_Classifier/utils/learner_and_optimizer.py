import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from fastai.vision.all import *
from fastcore.parallel import *
from fastai.optimizer import Optimizer
import os
from PIL import Image
import matplotlib.pyplot as plt

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# My own "DataLoader" class that load all the train and validation data
# for easy passing to my train_one_epoch function
class ModelData:
    # this is the class that saves all the data of training and validation set
    def __init__(self, train_x, train_y, valid_x, valid_y):
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y

# A linear activation layer
# 1. initialize params with gradient tracking
# 2. define loss, for classifications it's a 4-step,
## step 1 pred=x@weights + bias, the pred can any range, this is the activation
## step 2 sigmoid_pred = sigmoid(pred), the sigmoid_pred range (0,1), we read this as "the confidence of this image should be a labelled 1"
## step 3 loss = torch.where(y==1, 1-sigmoid_pred, sigmoid_pred)
## step 4 loss.mean(), report loss at this epoch
# 3. loss.backward() to get the gradient
# 4. step: (don't update gradients) params -= params.grad.item()*learning_rate
# 5. calc validation set loss, prediction and accuracy of this epoch, save it
# 6. set gradients to zero and go back to 2.
# plus, the functions should have lr and number of samples as changeable variables.
def init_params(dim, scale=1):
    # initialize parameters with random numbers draw from Standard Normal Distribution with desgined dim (tuple)
    # the weights should not be too large or too small, it will cause the
    return (torch.randn(dim)*scale).requires_grad_()

def linear1(x, weights, bias):
    # the simple linear regression activation
    return x@weights+bias

# For updating params with training set
def mnist_loss(pred, y):
    # for classification problem, prediction needs to be mapped to range (0,1)
    # this loss means: the average probability of making wrong predictions
    sig_pred = sigmoid(pred)
    return torch.where(y==1, 1-sig_pred, sig_pred).mean()
    # In PyTorch, torch.Size([]) means a 0-dim (scalar) tensor.
    # Many losses default to a reduction of "mean" (or "sum"), which collapses the batch and returns a single scalar value.

def calc_grad(model, x, y):
    # calculate gradients with the mnist_loss and activations
    # Warning: this includes the loss function, could not be used in MSE loss scenarios
    preds = model(x)
    loss = mnist_loss(preds, y)
    loss.backward()

# For calculating loss and accuracy of validation set
# my version to calculate accuracy metrics of validation set
def class_accuracy(pred, y):
    # class of pred is 1 or 0, compare with y which is also 1 or 0
    # my version:
    # this version have the accuracy to be always 0,
    # because pred is sigmoid from 0 to 1, while y is 0 and 1,
    # so y and pred are never going to be equal, of course the accuracy would be 0
    return 1.0*(y == pred).sum() / len(y)

# the textbook version of accuracy metrics
def batch_accuracy(pred, y):
    pred = sigmoid(pred)
    correct = (pred >=0.5).int() == y
    # if I am more careful, I would add the (1) dimension checks and (2) int/float type checks here
    return correct.float().mean()

def validate_epoch(model, valid_dl):
    # b stands for batch, every batch will contain some portion of the total training set
    accs = [batch_accuracy(model(xb),yb) for xb, yb in valid_dl]
    return round(torch.stack(accs).mean().item(),4)
    # tensor.item() returns the floats saved in the tensor, which could then be rounded by round() function

# my own function for doing equal sampling from different labels
# In SGD, I think we should avoid only sampling 1 label in one epoch
def stratified_splits_sample(y, n_samples):
    # given labels in y, randomly draw n_samples from each labels
    # 1. how many unique labels in y, what are the indexes for each label
    # 2. from each labels' list of indexes, draw n_samples, return a list
    # 3. combine the selected index's list of all labels and return it
    train_sample_indexes = {}
    for label in y.unique():
        idx = (y == label).all(dim=1).nonzero(as_tuple=True)[0]
        sampled_idx = idx[torch.randperm(idx.size(0), device=idx.device)[:n_samples]]
        train_sample_indexes[label.item()] = sampled_idx
    return train_sample_indexes

# my own function for GD with all training data in one iteration and calc the validation loss and accuracy
def train_one_epoch(model_func, mnist_loss, data, epoch, learning_rate, weights, bias):
    # model: could be linear or non-linear; data is the ModelData class
    train_pred = model_func(data.train_x, weights, bias)
    loss = mnist_loss(train_pred, data.train_y)
    loss.backward() # get gradients
    with torch.no_grad():
        weights -= learning_rate*weights.grad
        bias -= learning_rate*bias.grad
        valid_pred = model_func(data.valid_x, weights, bias)
        valid_loss = mnist_loss(valid_pred, data.valid_y)
        valid_accuracy = batch_accuracy(valid_pred, data.valid_y)
    # reset the gradients of weights and bias to Zero
    weights.grad.zero_()
    bias.grad.zero_()
    return {'epoch': epoch, 'train_loss': loss.item(),
            'valid_loss': valid_loss.item(),
            'valid_accuracy': valid_accuracy.item(),
            'learning_rate': learning_rate,
            }

# textbook
def train_one_epoch_by_batch(model_func, mnist_loss, epoch,
                             learning_rate, weights, bias, dl):
    for xb, yb in dl:
        train_pred = model_func(xb, weights, bias)
        loss = mnist_loss(train_pred, yb)
        loss.backward()
        with torch.no_grad():
            weights -= learning_rate*weights.grad
            bias -= learning_rate*bias.grad
        weights.grad.zero_()
        bias.grad.zero_()

def validate_one_epoch(model_func, valid_dl, weights, bias):
    with torch.no_grad():
        accs = [batch_accuracy(model_func(xb, weights, bias),yb) for xb, yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)

# textbook, the model_func here is a nn.Learner object that contains the weights and bias
def validate_epoch(model_func, valid_dl):
    with torch.no_grad():
        accs = [batch_accuracy(model_func(xb),yb) for xb, yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)

# my own SGD optimizer without momentum
class BasicOptimSGD():
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self, *args, **kwargs):
        # remove the gradients so it doesn't accumulate to next iteration
        for p in self.params: p.grad = None

    def step(self, *args, **kwargs):
        # updates params without tracking in autograd
        with torch.no_grad():
            for p in self.params:
                p -= self.lr*p.grad
        # for p in self.params: p.data -=self.lr*p.grad.data (this is Old style)

# the BasicOptimSGD cannot be directly called by the Learner object provided by fastai
# So I create a new one that inherit the fastai's Optimizer
class BasicOptimSGDInherit(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, lr)

    def step(self):
        for group in self.param_groups:          # each group is a dict
            for p in group['params']:            # the actual tensors
                if p.grad is not None:
                    p.data -= group['lr'] * p.grad.data

# a class that register the parameters (by myself) - not working
class all_params(nn.Module):
    def __init__(self, w1, w2, b1, b2):
        super(all_params, self).__init__()
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

# a class that register the parameters (by gpt)
class AllParams(nn.Module):
    """
    2-layer MLP params container: (x -> w1,b1 -> ReLU -> w2,b2).
    Creates leaf tensors with requires_grad=True and good inits.
    """
    def __init__(self, d_in, d_hidden, d_out, *, device=None, dtype=None, init="kaiming"):
        super().__init__()
        # allocate
        self.w1 = nn.Parameter(torch.empty(d_in, d_hidden, device=device, dtype=dtype))
        self.b1 = nn.Parameter(torch.zeros(d_hidden,      device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(d_hidden, d_out, device=device, dtype=dtype))
        self.b2 = nn.Parameter(torch.zeros(d_out,          device=device, dtype=dtype))
        # init
        self.reset_parameters(init)

    def reset_parameters(self, init="kaiming"):
        if init == "kaiming":
            nn.init.kaiming_uniform_(self.w1, a=0.0, mode="fan_in", nonlinearity="relu")
            nn.init.kaiming_uniform_(self.w2, a=0.0, mode="fan_in", nonlinearity="relu")
        elif init == "xavier":
            nn.init.xavier_uniform_(self.w1)
            nn.init.xavier_uniform_(self.w2)
        else:
            nn.init.normal_(self.w1, mean=0.0, std=0.02)
            nn.init.normal_(self.w2, mean=0.0, std=0.02)
        # biases are already zeros (often fine)

    @classmethod
    def from_tensors(cls, w1, b1, w2, b2):
        """
        If you *already* have tensors (even if they were created via ops),
        wrap them as fresh leaf Parameters safely.
        """
        d_in, d_hidden = w1.shape
        _, d_out       = w2.shape
        obj = cls.__new__(cls)          # bypass __init__ shape ctor
        nn.Module.__init__(obj)

        obj.w1 = nn.Parameter(w1.detach().clone(), requires_grad=True)
        obj.b1 = nn.Parameter(b1.detach().clone(), requires_grad=True)
        obj.w2 = nn.Parameter(w2.detach().clone(), requires_grad=True)
        obj.b2 = nn.Parameter(b2.detach().clone(), requires_grad=True)
        return obj

### Train one batch for multi-class 3 layer neural net
def three_layer_nn(linear_act, non_linear_act, params, xb):
    act1 = linear_act(xb, params.w1, params.b1)
    act2 = non_linear_act(act1)
    act3 = linear_act(act2, params.w2, params.b2)
    return act3

def train_one_batch_3layer(linear_act, non_linear_act, params, lr, xb, yb):
    # I think nn.Sequential is combining the 3 activations and their parameters into one, so the params can be passed onto optimizer as a whole
    act3 = three_layer_nn(linear_act, non_linear_act, params, xb)
    loss = F.cross_entropy(act3.as_subclass(torch.Tensor), yb.long().as_subclass(torch.Tensor), reduction='mean')
    loss.backward()
    # SGD update (in-place; do NOT rebind params.w*)
    with torch.no_grad():
        for p in (params.w1, params.b1, params.w2, params.b2):
            if p.grad is not None:              # guard in case some param didn't get a grad
                p.add_(p.grad, alpha=-lr)       # p -= lr * p.grad
        # clear grads for next step
        for p in (params.w1, params.b1, params.w2, params.b2):
            p.grad = None                       # or: if p.grad is not None: p.grad.zero_()
    return loss.item()

def validate_one_epoch_3layer(linear_act, non_linear_act, params, valid_dl):
    with torch.inference_mode():
        loss_agg = [F.cross_entropy(three_layer_nn(linear_act, non_linear_act, params, xb.view(xb.size(0), -1)).as_subclass(torch.Tensor),
                                    yb.long().as_subclass(torch.Tensor), reduction='mean')
                    for xb, yb in valid_dl]
    return round(torch.stack(loss_agg).mean().item(), 4)

def validate_one_epoch_accuracy(linear_act, non_linear_act, params, valid_dl, valid_ground_truth ):
    valid_predict_label = []
    with torch.inference_mode():
        for xb, yb in valid_dl:
            activation = three_layer_nn(linear_act, non_linear_act, params, xb.view(xb.size(0), -1))
            predict_label = torch.argmax(activation, dim=1).view(-1).long().as_subclass(torch.Tensor)
            valid_predict_label.extend(predict_label)
    valid_predict_label = torch.stack(valid_predict_label)
    (valid_predict_label == valid_ground_truth).sum().item()
    accuracy = round(1.0 * (valid_predict_label == valid_ground_truth).sum().item() / valid_ground_truth.shape[0], 4)
    return accuracy