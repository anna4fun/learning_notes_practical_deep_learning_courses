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
import utils.visualizers as vis
import torch.nn.functional as F
import torchviz as viz
from fastai.losses import CrossEntropyLossFlat
from datetime import datetime
importlib.reload(bmo)
importlib.reload(vis)

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
                   )
dls = dblock.dataloaders(train_files, batch_size=1024, shuffle=True) # mixture of batch
# Eyeball the dls
# number of batches
len(dls.train), len(dls.valid) # (46, 12)
# In total, the dataset has 48,000 training images and 12,000 validation images
len(dls.train_ds), len(dls.valid_ds) # (48000, 12000)

# use one batch for data exploration and routine checks
xb, yb = dls.train.one_batch()
xb[1].min(), xb[1].max() # (TensorImageBW(0., device='mps:0'), TensorImageBW(1., device='mps:0')) meaning the DataBlock applied the {'div': 255.0}
xb.shape, yb.shape # (torch.Size([128, 1, 28, 28]), torch.Size([128]))
# Does each batch contains a good mix of all classes?
yb
vals, cnts = yb.unique(return_counts=True)
vals # TensorCategory([0,     1,   2,   3,   4,   5,   6,   7,   8,   9], device='mps:0')
cnts # TensorCategory([105, 138, 101,  92,  96,  93,  97, 105,  88, 109], device='mps:0')
# looks like a good mix of labels
# put all the y labels in one tensor for accuracy metrics
valid_ground_truth = []
for xb, yb in dls.valid:
    valid_ground_truth.extend(yb.long().as_subclass(torch.Tensor))
valid_ground_truth=torch.stack(valid_ground_truth)
# see the data, x and label match
dls.show_batch(max_n=9, nrows=3, figsize=(6,6))
plt.show()

gc.collect()
torch.cuda.empty_cache()

### Initialize parameters\
w1 = bmo.init_params((28*28, 50), 1)
b1 = bmo.init_params((1, 50), 1)
w2 = bmo.init_params((50, 10), 1) # output dimension needs to be 10 to be mapped to the 10 digits
b2 = bmo.init_params((1, 10), 1)
w1.shape, b1.shape, w2.shape, b2.shape
# move your params to the same device
w1 = w1.to('mps');  b1 = b1.to('mps')
w2 = w2.to('mps');  b2 = b2.to('mps')
parameters = bmo.AllParams.from_tensors(w1, b1, w2, b2)

### compute all 3 layers of activation in one-shot
train_losses = {}
valid_losses = {}
valid_accuracy = {}
gc.collect()
torch.cuda.empty_cache()
# Print the formatted current time
print("Current Time =", datetime.now().strftime("%H:%M:%S")) # 13:02:34

for epoch in range(11, 20):
    for batch, (xb, yb) in enumerate(dls.train, start=1):
        current_step = str(epoch) + '-' + str(batch)
        train_losses[current_step] = bmo.train_one_batch_3layer(bmo.linear1, nn.LeakyReLU(0.1), parameters, lr=0.1, xb=xb.view(xb.size(0), -1), yb=yb)
    # validate per epoch
    valid_losses[str(epoch)] = bmo.validate_one_epoch_3layer(bmo.linear1, nn.LeakyReLU(0.1), parameters, dls.valid)
    valid_accuracy[str(epoch)] = bmo.validate_one_epoch_accuracy(bmo.linear1, nn.LeakyReLU(0.1), parameters, dls.valid,valid_ground_truth)

# Print the formatted current time
print("Current Time =", datetime.now().strftime("%H:%M:%S")) # Current Time = 13:05:32

# Visualize results
vis.plot_loss_and_accuracy(valid_losses, valid_accuracy,
                           savepath="MNIST_Handwritten_Digits_Classifier/results/10-digits-training-loss_and_accuracy_curve_02.png")

vis.plot_losses(train_losses, valid_losses)
plt.savefig("MNIST_Handwritten_Digits_Classifier/results/10-digits-training-loss_curve_01.png", dpi=200, bbox_inches="tight")



