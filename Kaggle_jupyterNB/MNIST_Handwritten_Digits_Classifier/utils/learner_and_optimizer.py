import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fastai.vision.all import *
from fastcore.parallel import *
import os
from PIL import Image
import matplotlib.pyplot as plt

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# helper functions for showing an image
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

class ModelData:
    # this is the class that saves all the data of training and validation set
    def __init__(self, train_x, train_y, valid_x, valid_y):
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y

# A linear layer
# 1. initialize params with gradient tracking
# 2. define loss, for classifications it's a 4-step,
## step 1 pred=x@weights + bias, the pred can any range
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
    return (torch.randn(dim)*scale).requires_grad_()

def linear1(x, weights, bias):
    return x@weights+bias

def mnist_loss(pred, y):
    sig_pred = sigmoid(pred)
    return torch.where(y==1, 1-sig_pred, sig_pred).mean()

def calc_grad(model, x, y):
    preds = model(x)
    loss = mnist_loss(preds, y) # the average probability of making wrong predictions
    loss.backward()

def class_accuracy(pred, y):
    # class of pred is 1 or 0, compare with y which is also 1 or 0
    # my version:
    # this version have the accuracy to be always 0,
    # because pred is sigmoid from 0 to 1, while y is 0 and 1,
    # so y and pred are never going to be equal, of course the accuracy would be 0
    return 1.0*(y == pred).sum() / len(y)


def batch_accuracy(pred, y):
    pred = sigmoid(pred)
    correct = (pred >=0.5).int() == y
    return correct.float().mean()

def validate_epoch(model, valid_dl):
    # b stands for batch, every batch will contains some portion of the total training set
    accs = [batch_accuracy(model(xb),yb) for xb, yb in valid_dl]
    return round(torch.stack(accs).mean().item(),4)


# TODO: stratified sampling from the 2 class
def stratified_splits_sample(y, n_samples):
    train_sample_indexes = torch.randint(0, y.shape[0], size = (n_samples,))
    return train_sample_indexes

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






