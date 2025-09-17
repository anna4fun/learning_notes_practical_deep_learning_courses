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
# see one image
fours = (train_files / "4").ls().sorted()
sample4 = tensor(Image.open(fours[20])) # torch.Size([28, 28])
sample4.min(), sample4.max() #(tensor(0, dtype=torch.uint8), tensor(254, dtype=torch.uint8))
bmo.show_image(sample4)

# load the training data into a Data Block
dblock = DataBlock(blocks = (ImageBlock(cls=PILImageBW), CategoryBlock), # PILImageBW means black and white (RGB channel = 1)
                   get_items=get_image_files,  # finds all images recursively
                   splitter=RandomSplitter(0.2),
                   get_y=parent_label,  # folder name (0..9) becomes the label)
                   item_tfms=Resize(28)
                   )
dblock.summary(train_files)
# Setting up before_batch: Pipeline:
# Setting up after_batch: Pipeline: IntToFloatTensor -- {'div': 255.0, 'div_mask': 1}
# Building one batch
# Applying item_tfms to the first sample:
#   Pipeline: Resize -- {'size': (28, 28), 'method': 'crop', 'pad_mode': 'reflection', 'resamples': (<Resampling.BILINEAR: 2>, <Resampling.NEAREST: 0>), 'p': 1.0} -> ToTensor
#     applying ToTensor gives
#       (TensorImageBW of size 1x28x28, TensorCategory(4))
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
[dls.vocab[i] for i in yb]

# Linear activation
xb = xb.view(-1, 28*28) # flatten xb
xb.shape, yb.shape # (torch.Size([64, 784]), torch.Size([64]))
# act_layer_1 = nn.Linear(in_features=28*28, out_features=50)
w1 = bmo.init_params((28*28, 50), 1)
b1 = bmo.init_params((1, 50), 1)
w2 = bmo.init_params((50, 10), 1) # output dimension needs to be 10 to be mapped to the 10 digits
b2 = bmo.init_params((1, 10), 1)
w1.shape, b1.shape, w2.shape, b2.shape

# RuntimeError: Tensor for argument #2 'mat2' is on CPU, but expected it to be on GPU (while checking arguments for mm)
# This error means your tensors are on different devices. Your xb (data) is on GPU (CUDA/MPS), while w1 (weights) is on CPU — or vice-versa. For matrix multiply, both must be on the same device.
print(xb.device)  # mps:0
print(w1.device)  # cpu
print(b1.device)  # cpu
# move your params to xb's device
w1 = w1.to(xb.device);  b1 = b1.to(xb.device)
w2 = w2.to(xb.device);  b2 = b2.to(xb.device)

### compute all 3 layers of activation in one-shot
gc.collect()
torch.cuda.empty_cache()
# layer 1 linear
act1 = bmo.linear1(xb, w1, b1)
act1.shape # torch.Size([64, 50])
act1[0]
# layer 2 ReLu
act2 = torch.relu(act1)
act2.shape # torch.Size([64, 50])
act2[0]
# layer 3 linear
act3 = bmo.linear1(act2, w2, b2)
act3.shape # torch.Size([64, 10])
act3[63]
# TensorImageBW([ 39.9587, -40.7243, -20.7452, -29.5299,  27.9315,
#                  -88.4787, 0.8585, -49.0575,  68.3239, -19.4959], device='mps:0',
#               grad_fn=<AliasBackward0>)
pred = act3
y = yb
# experiment softmax + nll_loss (without taking log)
sig_pred = torch.softmax(pred, dim=1).as_subclass(torch.Tensor)  # change from <class 'fastai.torch_core.TensorImageBW'> into a Tensor
prob_mistake = F.nll_loss(sig_pred, y, reduction='none')
sig_pred.shape
sig_pred[63]
# TensorImageBW([9.3994e-14, 0.0000e+00, 0.0000e+00, 7.1491e-11, 5.9984e-07,
#                0.0000e+00, 3.4423e-33, 0.0000e+00, 1.0000e+00, 1.5965e-37],
#               device='mps:0', grad_fn=<AliasBackward0>)
# Interesting: act3[63]'s 2 biggest values are 39.9587 and 68.3239,
# these 2 values become 9.3994e-14 and 1, the difference is Huge!
yb[63] # category = 6, meaning we will pick sig_pred[63][6] = 3.4423e-33
prob_mistake[63] # TensorCategory(-3.4423e-33, device='mps:0', grad_fn=<AliasBackward0>), correct
prob_mistake
torch.log(prob_mistake)
# TensorCategory([nan, -inf, nan, -inf, nan, nan, nan, nan, nan, nan, -inf, -inf, nan, nan, nan, -inf, -inf, -inf, -inf, -inf, nan,
#                 nan, -inf, nan, nan, -inf, -inf, nan, nan, nan, nan, nan, -inf, nan, nan, -inf, nan, nan, nan, -inf, -inf, nan,
#                 nan, nan, nan, nan, nan, -inf, nan, nan, nan, nan, nan, -inf, nan, nan, nan, nan, nan, nan, nan, nan, -inf,
#                 nan], device='mps:0', grad_fn=<AliasBackward0>)
## Too many nan,it's impossible to take mean directly

ce_loss = F.cross_entropy(act3.as_subclass(torch.Tensor), yb.long().as_subclass(torch.Tensor), reduction='none')   # internally does log_softmax + NLL, numerically stable
ce_loss.shape # torch.Size([64])
ce_loss
# tensor([-0.0000e+00, 1.0481e+02, 9.7546e-01, 9.8476e+01, 1.8429e+01, 2.9443e+01,
#         2.8128e+00, 3.8692e+01, 3.0248e+01, 5.1674e+01, 1.1006e+02, 1.5478e+02,
#         2.3842e-07, 8.5601e+01, 1.2775e+01, 1.4577e+02, 1.5440e+02, 1.3474e+02,
#         1.7896e+02, 8.7743e+01, 1.0623e+01, 4.2291e+01, 1.7303e+02, 5.1827e+01,
#         4.3621e+01, 1.4877e+02, 9.9389e+01, 1.4097e+01, 2.8677e+00, 4.6592e+01,
#         5.3231e+01, 2.4461e+01, 1.1817e+02, 6.2818e+00, -0.0000e+00, 1.0948e+02,
#         6.0303e+01, 7.4263e-01, 7.8284e+01, 1.0824e+02, 1.0603e+02, 6.5716e+01,
#         6.7855e+01, 7.6141e+00, 3.1466e+01, 2.9084e+01, 2.0258e-02, 2.0249e+02,
#         4.9887e+01, 3.6297e+01, 5.2213e+01, 7.9661e+01, -0.0000e+00, 1.4971e+02,
#         1.1340e+01, 3.8223e+01, 3.8422e+01, 5.4343e+01, 6.9045e+01, 1.0548e+00,
#         1.9282e+01, 1.9565e+01, 9.5920e+01, 7.4749e+01], device='mps:0',
#        grad_fn=<NllLossBackward0>)
ce_loss.mean() # tensor(61.4479, device='mps:0', grad_fn=<MeanBackward0>)
ce_loss_final = F.cross_entropy(act3.as_subclass(torch.Tensor), yb.long().as_subclass(torch.Tensor), reduction='mean')
ce_loss_final # tensor(61.4479, device='mps:0', grad_fn=<NllLossBackward0>)

ce_loss_final.backward

# name your params for labeling
param_map = {
    "W1": w1, "b1": b1,
    "W2": w2, "b2": b2,
}
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"
g = viz.make_dot(ce_loss_final, params=param_map)
g.render("MNIST_Handwritten_Digits_Classifier/3_layer_NN_computation_graph", format="png")

## Problem: all predicted labels are 8, why?
## because every row in the final activation are the same
xb,yb = dls.valid.one_batch()
activation = bmo.three_layer_nn(bmo.linear1, torch.relu, parameters, x2)
activation.shape
valid_label = torch.argmax(activation, dim=1).view(-1)
valid_label.long().as_subclass(torch.Tensor)
yb.long().as_subclass(torch.Tensor)
# is xb[0] and xb[1] the same?
xb[0]
xb[-1]
xb.view(-1, 28*28)
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


## ChatGPT's solution to my 3-layer MLP
# it's faster than my own solution
# params as nn.Parameter so optimizers work correctly
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
w1 = nn.Parameter(torch.empty(784, 50, device=device))
b1 = nn.Parameter(torch.empty(50, device=device))
w2 = nn.Parameter(torch.empty(50, 10, device=device))
b2 = nn.Parameter(torch.empty(10, device=device))

nn.init.kaiming_normal_(w1, nonlinearity='relu'); nn.init.constant_(b1, 0.01)
nn.init.kaiming_normal_(w2, nonlinearity='relu'); nn.init.zeros_(b2)

params = [w1, b1, w2, b2]
opt = torch.optim.Adam(params, lr=1e-3)

def forward(x):
    x = x.view(x.size(0), -1)
    x = (x - 0.1307) / 0.3081   # normalize
    h = torch.nn.functional.leaky_relu(x @ w1 + b1, 0.01)
    return h @ w2 + b2          # logits

for epoch in range(5, 10):
    # ---- train ----
    for xb, yb in dls.train:
        xb, yb = xb.to(device), yb.to(device).long()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(forward(xb).as_subclass(torch.Tensor), yb.as_subclass(torch.Tensor))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

    # ---- validate ----
    correct = total = 0
    with torch.inference_mode():
        for xb, yb in dls.valid:
            xb, yb = xb.to(device), yb.to(device).long()
            pred = forward(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    acc = correct/total
    print(f"epoch {epoch+1} acc={acc:.4f}")

