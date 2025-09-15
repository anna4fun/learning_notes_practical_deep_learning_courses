import torch.nn
from fastai.vision.all import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from PIL import Image
import gc
import importlib
import utils.learner_and_optimizer as bmo
import torch.nn.functional as F
import torchviz as viz
from fastai.losses import CrossEntropyLossFlat
importlib.reload(bmo)

## Differences between binary classification and multi-label classification problems
# 1. loss function needs to be cross-entropy: activation -> softmax -> negative log-likelihood
# 2. (not needed) targets: one-hot-encoding for 0-9
# 3. implement a 3-layer NN my self. 1st layer 786x

# ========================================================================================
# import MNIST dataset
# URLs is an object that stores all the URLs of datasets, in this tutorial we are going to use the  MNIST_SAMPLE and MINIST_TINY
# let's define lists of path to images which will be used for data loading
# path = untar_data(URLs.MNIST)
path = Path.home() / ".fastai/data/mnist_png"
path.ls()
train_files = path/ "training/"

# load the training data into a Data Block
dblock = DataBlock(blocks = (ImageBlock(cls=PILImageBW), CategoryBlock), # PILImageBW means black and white (RGB channel = 1)
                   get_items=get_image_files,  # finds all images recursively
                   splitter=RandomSplitter(0.2),
                   get_y=parent_label,  # folder name (0..9) becomes the label)
                   item_tfms=Resize(28),
                   batch_tfms=Resize(128),
                   )
dls = dblock.dataloaders(train_files, batch_size=128)
# Eyeball the dls
# number of batches
len(dls.train), len(dls.valid) # (750, 188)
# In total, the dataset has 48,000 training images and 12,000 validation images
len(dls.train_ds), len(dls.valid_ds) # (48000, 12000)

# use one batch for experimentation
xb, yb = dls.one_batch()
xb[1].min(), xb[1].max() # (TensorImageBW(0., device='mps:0'), TensorImageBW(1., device='mps:0')) meaning the DataBlock applied the {'div': 255.0}
xb.shape, yb.shape # (torch.Size([64, 3,  28, 28]), torch.Size([64]))
dls.show_batch(max_n=9, nrows=3, figsize=(6,6))
plt.show()
### Initialize parameters\
w1 = bmo.init_params((28*28, 50), 1)
b1 = bmo.init_params((1, 50), 1)
w2 = bmo.init_params((50, 10), 1) # output dimension needs to be 10 to be mapped to the 10 digits
b2 = bmo.init_params((1, 10), 1)
w1.shape, b1.shape, w2.shape, b2.shape
# move your params to the same device
w1 = w1.to('mps');  b1 = b1.to('mps')
w2 = w2.to('mps');  b2 = b2.to('mps')
w1.device, b1.device # (device(type='mps', index=0), device(type='mps', index=0), device(type='mps', index=0))


parameters = bmo.AllParams.from_tensors(w1, b1, w2, b2)
parameters.w1.shape
parameters.w2.device
parameters.w2.shape
parameters.b2.shape

### compute all 3 layers of activation in one-shot
gc.collect()
torch.cuda.empty_cache()

batch = 0
train_losses = {}
valid_losses = {}
for batch, (xb, yb) in enumerate(dls.train, start=1):
    if batch > 200:
        break
    train_losses[str(batch)] = bmo.train_one_epoch_3layer(bmo.linear1, torch.relu, parameters, lr=0.1, xb=xb.view(-1, 28*28), yb=yb)
    valid_losses[str(batch)] = bmo.validate_one_epoch_3layer(bmo.linear1, torch.relu, parameters, dls.valid)


# running for too long (>40mins) so I interpted it
batch # 201
train_losses
valid_losses

valid_ground_truth = []
for xb, yb in dls.valid:
    valid_ground_truth.extend(yb.long().as_subclass(torch.Tensor))
valid_ground_truth=torch.stack(valid_ground_truth)

valid_predict_label = []
with torch.inference_mode():
    for xb, yb in dls.valid:
        activation = bmo.three_layer_nn(bmo.linear1, torch.nn.LeakyReLU(0.1), parameters, xb.view(-1, 28 * 28))
        predict_label = torch.argmax(activation, dim=1).view(-1).long().as_subclass(torch.Tensor)
        valid_predict_label.extend(predict_label)


valid_predict_label= torch.stack(valid_predict_label)
(valid_predict_label==valid_ground_truth).sum().item()
valid_ground_truth.shape # torch.Size([12000])
acc = round(1.0*(valid_predict_label==valid_ground_truth).sum().item()/valid_ground_truth.shape[0],4)
## batch-size = 128
# with training on 19 batches and lr=0.1
acc # 0.3804
# with training on 101 batches and lr=0.1
acc # 0.6179
# 201 batches
# train loss : {'1': 2.5298, '2': 3.3488} strange, why would the first batch have loss 2.5?
acc # 0.5353

## batch-size =  64
# with training on 11 batches and learning rate= 0.1
acc #0.3895
# training on 101 batches and learning rate =0.1
acc #0.6867
# training on 201 batches and learning rate =0.1
acc #0.7509
# training on 301
acc # 0.4021 it regressed! It could be because the batch size is 64 too small

## Problem: all predicted labels are 8, why?
## because every row in the final activation are the same
xb,yb = dls.valid.one_batch()
activation = bmo.three_layer_nn(bmo.linear1, torch.relu, parameters, xb.view(-1, 28 * 28))
valid_label = torch.argmax(activation, dim=1).view(-1)
valid_label.long().as_subclass(torch.Tensor)
yb.long().as_subclass(torch.Tensor)
# is xb[0] and xb[1] the same?
xb[0]
xb[-1]
torch.equal(xb[0], xb[1]) # False
torch.equal(activation[0], activation[1]) # True
torch.equal(w1[0], w1[1]) # False
w1.shape
xb.shape, yb.shape
act1 = bmo.linear1(xb.view(-1, 28*28), parameters.w1, parameters.b1)
torch.equal(act1[0], act1[1]) # False
act1[0] # all negative
act1[1]
act2 = torch.relu(act1)
torch.equal(act2[0], act2[-1]) # True so this is creating the equals
act2[0] # all zero
act2[-1] # all zero
# that’s the classic “dying ReLU” failure mode:
# Too high LR can shove the first layer so negative it never recovers. change LR from 10 to 0.1
# Use leaky Relu.

bmo.plot_losses(train_losses, valid_losses)
plt.savefig("MNIST_Handwritten_Digits_Classifier/results/10-digits-training-loss_curve_01.png", dpi=200, bbox_inches="tight")
valid_predict_label # all of them are predicted to be 8, which is not good
# simple_net = nn.Sequential(nn.Linear(28*28,50),
#                            nn.ReLU(),
#                            nn.Linear(50,10))
# nnlearn = Learner(dls, simple_net, opt_func=SGD(lr=10),
#                   loss_func=nn.CrossEntropyLoss, metrics=)


