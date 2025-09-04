import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fastai.vision.all import *
from fastcore.parallel import *
from fastai.optimizer import Optimizer
import os
from PIL import Image
import matplotlib.pyplot as plt

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Helper functions for showing an image
def print_image_as_pixel(image_tensor, decimal=2):
    # only works in Jupyter env
    tdf = pd.DataFrame(image_tensor)
    styler = tdf.style.format(
        lambda x: f"{x:.{decimal}f}".rstrip('0').rstrip('.') if isinstance(x, float) else x
    )
    return (styler
            .set_properties(**{'font-size':'7pt'})
            .background_gradient(cmap='Greys'))

def show_image(image_tensor):
    # works in Python console
    arr = getattr(image_tensor, 'detach', lambda: image_tensor)().cpu().numpy() \
          if hasattr(image_tensor, 'cpu') else np.asarray(image_tensor)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            r, g, b = arr[0], arr[1], arr[2]
            arr = 0.2989*r + 0.5870*g + 0.1140*b
    plt.imshow(arr, cmap='gray')
    plt.axis('off')
    plt.show()

def file_size(image):
    return PILImage.create(image).size

# Plotting the training stats, train loss, valid loss and valid accuracy
def plot_epoch_stats(plt, epoch_num, lr, learner_name, learner):
    train_loss = L(learner.recorder.values).itemgot(0)
    valid_loss = L(learner.recorder.values).itemgot(1)
    valid_acc = L(learner.recorder.values).itemgot(2)
    # Find best epoch (based on highest validation accuracy)
    best_epoch = int(np.argmax(valid_acc))
    best_acc = valid_acc[best_epoch]

    # Create 3 horizontal subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(train_loss)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    axes[1].plot(valid_loss)
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].axvline(best_epoch, color="red", linestyle="--", label=f"Best Epoch {best_epoch}")
    axes[1].grid(True)

    axes[2].plot(valid_acc)
    axes[2].set_title("Validation Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].axvline(best_epoch, color="red", linestyle="--")
    axes[2].plot(best_epoch, best_acc, "ro", label=f"Best Acc {best_acc:.2f}")
    axes[2].legend()
    axes[2].grid(True)

    # Add a global figure title
    fig.suptitle(f"{learner_name} Training Results (Epochs={epoch_num}, Learning Rate={lr})", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return plt


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

