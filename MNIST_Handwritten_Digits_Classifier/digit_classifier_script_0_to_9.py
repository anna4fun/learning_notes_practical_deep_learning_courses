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
# # number of batches
# len(dls.train), len(dls.valid) # (46, 12)
# # In total, the dataset has 48,000 training images and 12,000 validation images
# len(dls.train_ds), len(dls.valid_ds) # (48000, 12000)
#
# # use one batch for data exploration and routine checks
# xb, yb = dls.train.one_batch()
# xb[1].min(), xb[1].max() # (TensorImageBW(0., device='mps:0'), TensorImageBW(1., device='mps:0')) meaning the DataBlock applied the {'div': 255.0}
# xb.shape, yb.shape # (torch.Size([128, 1, 28, 28]), torch.Size([128]))
# # Does each batch contains a good mix of all classes?
# yb
# vals, cnts = yb.unique(return_counts=True)
# vals # TensorCategory([0,     1,   2,   3,   4,   5,   6,   7,   8,   9], device='mps:0')
# cnts # TensorCategory([105, 138, 101,  92,  96,  93,  97, 105,  88, 109], device='mps:0')
# # looks like a good mix of labels
# put all the y labels in one tensor for accuracy metrics
valid_ground_truth = []
for xb, yb in dls.valid:
    valid_ground_truth.extend(yb.long().as_subclass(torch.Tensor))
valid_ground_truth=torch.stack(valid_ground_truth)
# see the data, x and label match
# dls.show_batch(max_n=9, nrows=3, figsize=(6,6))
# plt.show()

### Initialize parameters
## Use Standard Normal initialization
# w1 = bmo.init_params((28*28, 50), 1)
# b1 = bmo.init_params((1, 50), 1)
# w2 = bmo.init_params((50, 10), 1) # output dimension needs to be 10 to be mapped to the 10 digits
# b2 = bmo.init_params((1, 10), 1)
# w1.shape, b1.shape, w2.shape, b2.shape
# # move your params to the same device
# w1 = w1.to('mps');  b1 = b1.to('mps')
# w2 = w2.to('mps');  b2 = b2.to('mps')
# parameters = bmo.AllParams.from_tensors(w1, b1, w2, b2)
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")

## Use Kaiming initialization
# hidden = 50
# hidden = 128
hidden = 256
w1 = nn.Parameter(torch.empty(784, hidden, device=device))
b1 = nn.Parameter(torch.empty(hidden, device=device))
w2 = nn.Parameter(torch.empty(hidden, 10, device=device))
b2 = nn.Parameter(torch.empty(10, device=device))
nn.init.kaiming_normal_(w1, nonlinearity='relu'); nn.init.constant_(b1, 0.01)
nn.init.kaiming_normal_(w2, nonlinearity='relu'); nn.init.zeros_(b2)
parameters = bmo.AllParams.from_tensors(w1, b1, w2, b2)

### A 3-layered Neural Network (MLP)
train_losses = {}
valid_losses = {}
valid_accuracy = {}
gc.collect()
torch.cuda.empty_cache()
# Print the formatted current time
print("Current Time =", datetime.now().strftime("%H:%M:%S")) # 14:00:32

lr = 0.1 # epoch 0-19, max accuracy 0.948 plateu
lr = 0.01
# decaying the learning rates as epochs increase
# def lr_factor(t):  # t = global step
#     return 0.5 ** (t // 2)

for epoch in range(20, 24):
    for batch, (xb, yb) in enumerate(dls.train, start=1):
        current_step = str(epoch) + '-' + str(batch)
        train_losses[current_step] = bmo.train_one_batch_3layer(bmo.linear1, nn.LeakyReLU(0.1),
                                                                parameters, lr=lr,
                                                                xb=xb.view(xb.size(0), -1), yb=yb)
    # validate per epoch
    # valid_losses[str(epoch)] = bmo.validate_one_epoch_3layer(bmo.linear1, nn.LeakyReLU(0.1), parameters, dls.valid)
    # valid_accuracy[str(epoch)] = bmo.validate_one_epoch_accuracy(bmo.linear1, nn.LeakyReLU(0.1),
    #                                                              parameters,
    #                                                              dls.valid,valid_ground_truth)
    ## Time to see which images are misclassified
    if epoch in (0, 1, 5, 8, 14, 16, 19, 20, 23):
        bmo.evaluate_validation_and_show_misclassified(dls, show_n=16, ncols=4, params=parameters,
                              save_dir = "MNIST_Handwritten_Digits_Classifier/results/3_layer_NN/set4/",
                              mis_fname=str(epoch) + "_misclassified.png",
                              cm_fname=str(epoch) + "_confusion_matrix.png")


print("Current Time =", datetime.now().strftime("%H:%M:%S")) # Current Time = 14:09:37
valid_accuracy
# Visualize results
vis.plot_loss_and_accuracy(valid_losses, valid_accuracy,
                           savepath="MNIST_Handwritten_Digits_Classifier/results/3_layer_NN/set4-10-digits-training-loss_and_accuracy_curve.png")
plt = vis.plot_losses(train_losses, valid_losses)
plt.savefig("MNIST_Handwritten_Digits_Classifier/results/3_layer_NN/set4-10-digits-training-loss_curve.png", dpi=200, bbox_inches="tight")



# Run the evaluation, show some mistakes and the confusion matrix
eval = bmo.evaluate_validation_and_show_misclassified(dls, show_n=16, ncols=4, params=parameters
                                                      ,save_dir="MNIST_Handwritten_Digits_Classifier/results/3_layer_NN/")





