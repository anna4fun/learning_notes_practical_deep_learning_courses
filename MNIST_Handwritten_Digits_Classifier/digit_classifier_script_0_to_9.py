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
                   item_tfms=Resize(28)
                   )
dls = dblock.dataloaders(train_files)
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
w1 = w1.to(xb.device);  b1 = b1.to(xb.device)
w2 = w2.to(xb.device);  b2 = b2.to(xb.device)
xb.device, w1.device, b1.device # (device(type='mps', index=0), device(type='mps', index=0), device(type='mps', index=0))


parameters = bmo.AllParams.from_tensors(w1, b1, w2, b2)
parameters.w1.shape
parameters.w2.device
parameters.w2.shape
parameters.b2.shape

### compute all 3 layers of activation in one-shot
gc.collect()
torch.cuda.empty_cache()
#
xb = xb.view(-1, 28*28) # flatten xb
xb.shape, yb.shape # (torch.Size([64, 784]), torch.Size([64]))
lb = bmo.train_one_epoch_3layer(bmo.linear1, torch.relu, parameters, lr=10, xb=xb, yb=yb)
lb # 72.57
batch = 0
train_losses = {}
valid_losses = {}
for xb, yb in dls.train:
    batch += 1
    train_losses[str(batch)] = bmo.train_one_epoch_3layer(bmo.linear1, torch.relu, parameters, lr=10, xb=xb.view(-1, 28*28), yb=yb)
    valid_losses[str(batch)] = bmo.validate_one_epoch_3layer(bmo.linear1, torch.relu, parameters, dls.valid)

# running for too long (>40mins) so I interpted it
batch # 412
train_losses
valid_losses
valid_predict_label = []
valid_ground_truth = []
with torch.no_grad():
    for xb, yb in dls.valid:
        valid_label = bmo.three_layer_nn(bmo.linear1, torch.relu, parameters, xb.view(-1, 28 * 28)).argmax(dim=1).view(-1)
        valid_predict_label.extend(valid_label)
        valid_ground_truth.extend(yb)

bmo.plot_losses(train_losses, valid_losses)
plt.savefig("MNIST_Handwritten_Digits_Classifier/results/10-digits-training-loss_curve_01.png", dpi=200, bbox_inches="tight")
valid_predict_label # all of them are predicted to be 8, which is not good
# simple_net = nn.Sequential(nn.Linear(28*28,50),
#                            nn.ReLU(),
#                            nn.Linear(50,10))
# nnlearn = Learner(dls, simple_net, opt_func=SGD(lr=10),
#                   loss_func=nn.CrossEntropyLoss, metrics=)


