import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fastai.vision.all import *
from fastcore.parallel import *
import os
from PIL import Image
import utils.learner_and_optimizer as bmo
import gc
import importlib
from matplotlib import pyplot as plt

importlib.reload(bmo)


# ========================================================================================
# import MNIST dataset
# URLs is an object that stores all the URLs of datasets, in this tutorial we are going to use the  MNIST_SAMPLE and MINIST_TINY
# let's define lists of path to images which will be used for data loading
#path = untar_data(URLs.MNIST_SAMPLE) # Path('/Users/lcjh/.fastai/data/mnist_sample')
path = Path.home()/ "fastai/data/mnist_sample"
sevens = (path/'train'/'7').ls().sorted()
threes = (path/'train'/'3').ls().sorted()
valid_sevens = (path/'valid'/'7').ls().sorted()
valid_threes = (path/'valid'/'3').ls().sorted()

# ========================================================================================
## Prep 1: Sanity Checks
# How many images are there?
print(len(threes)) # 6131
print(len(sevens)) # 6265

# randomly pick one images from each set to eyeball the validity of the image
m3t = tensor(Image.open(threes[np.random.randint(len(threes))]))
# print_image_as_pixel(m3t)
show_image(m3t)
m7t = tensor(Image.open(sevens[np.random.randint(len(sevens))]))
# print_image_as_pixel(m7t)
show_image(m7t)

# Are all images of the same size? pass the file path to the get_image_files() func
# same size training data is important to make GPU happy
train_files = get_image_files(path/'train')
train_file_size = parallel(bmo.file_size, train_files, n_workers=10)
file_sizes = pd.Series(train_file_size).value_counts() # (28, 28)    12396

## Prep 2: The data structure of organizing training and validation data
# basic unit: a tensor
# Loading the training and validation images into DataSets
# Use stack to turn nested list into a rank-3 tensor. Because the nested list still not easy to iterate and average each images.
# Noted that GPU computing default datatype is *Float*
# GPUs (especially with CUDA/cuDNN) are optimized for floating-point math:
# FP32 (float32) is the default training datatype.
# Many models now use FP16 (half precision) or bfloat16 for faster matrix multiplies.
# The GPU doesn’t really “prefer” big or small numbers — what matters is keeping them in a numerically stable range that avoids overflow or underflow.
# So here, the original pixel values are int range up to 256, let's downsize it by dividing it with 256
# this will (1) make it float (2) avoid overflow after the linear activation
stacked_three = torch.stack([tensor(Image.open(o))/256 for o in threes])
stacked_seven = torch.stack([tensor(Image.open(o))/256 for o in sevens])
stacked_three_valid = torch.stack([tensor(Image.open(o))/256 for o in valid_threes])
stacked_seven_valid = torch.stack([tensor(Image.open(o))/256 for o in valid_sevens])
stacked_three.shape, stacked_seven.shape, stacked_three_valid.shape, stacked_seven_valid.shape

# I want the train_x to be a 2-D matrix of (6131+6265, 784)
train_x = torch.cat([stacked_three, stacked_seven]).view(-1, 28*28)
train_y = tensor([1]*len(stacked_three)+[0]*len(stacked_seven)).unsqueeze(1)
train_y[stacked_three.shape[0]-1]
valid_x = torch.cat([stacked_three_valid, stacked_seven_valid]).view(-1, 28*28)
valid_x[0]
valid_y = tensor([1]*stacked_three_valid.shape[0]+[0]*stacked_seven_valid.shape[0]).unsqueeze(1)
valid_y[stacked_three_valid.shape[0]]

# Packing all the data into one model_data to be passed into training function
model_data = bmo.ModelData(train_x, train_y, valid_x, valid_y)

# ========================================================================================
## Part 1: Linear Regression on All Training Data
training_epoch_results = []
# for learning_rate in [100, 150, 200]:
for learning_rate in [10, 15, 20]:
    gc.collect()
    torch.cuda.empty_cache()
    weights = bmo.init_params((train_x.shape[1], 1))
    bias = bmo.init_params(1)
    for i in range(20):
        results = bmo.train_one_epoch(bmo.linear1, bmo.mnist_loss, model_data, epoch=i,
                                      learning_rate=learning_rate, weights=weights, bias=bias)
        training_epoch_results.append(results)

results_df = pd.DataFrame(training_epoch_results)
results_df
# With learning_rate = 0.1, accuracy and loss don't change much
# weights.min(), weights.max()
# (tensor(-38.4075, grad_fn=<MinBackward1>), tensor(33.0707, grad_fn=<MaxBackward1>))
# weights.grad.min(), weights.grad.max()
# (tensor(-0.0025), tensor(0.0010))
# the gradients is so tiny compared with the weights' scale
# with small lr = 0.1, the weights doesn't change much
# after updating the lr = 100, the accuracy goes from 0.5 to 0.87 within 20 epoch
# I experimented 3 learning_rates 100, 150, 200,
# found out that learning_rate = 150 able to achieve the highest accuracy 0.92 at epoch=20
# I further downsize the weights from scale = 10 into scale = 1,
# then use learning_rate at [10,15,20] gives similar results.

# Note: All results are saved under the `results` folder
cwd = os.getcwd()
output_dir = cwd + '/MNIST_Handwritten_Digits_Classifier/results/v2-digit_classifier_results_w_div256_scale_pixel_weights_scale1.csv'
results_df.to_csv(output_dir, index=False)

# Thinking
# 1. I think the small scale of gradients is caused by the training dataset's pixels all divided by 256, aka scale of x determine the scale of gradients
w0=bmo.init_params((train_x.shape[1],1))
b0=bmo.init_params(1)
# shape checking, don't make pred in the shape of (786,10) LOL
w0.shape, b0.shape # (torch.Size([784, 1]), torch.Size([1]))
pred = bmo.linear1(model_data.valid_x, w0, b0)
pred.shape
acc = bmo.batch_accuracy(pred, model_data.valid_y)
acc, acc.shape
loss = bmo.mnist_loss(pred, model_data.valid_y)
loss.shape, loss.item() # (torch.Size([]), 0.5325248837471008)
loss.backward()
with torch.no_grad():
    print(w0.grad.min(), w0.grad.max()) # tensor(-0.0333) tensor(0.0060)
    print(w0.min(), w0.max()) # tensor(-3.2619) tensor(2.7059)
## with div 256 training data, initial weights takes scale=1
# print(w0.min(), w0.max()) : tensor(-3.2767) tensor(3.1964)
# print(w0.grad.min(), w0.grad.max()): tensor(-0.0208) tensor(0.0131)

## with original big training data, initial weights takes scale=1
# print(w0.min(), w0.max()) # tensor(-3.2619) tensor(2.7059)
# print(w0.grad.min(), w0.grad.max()) # tensor(-0.0333) tensor(0.0060)

## with original big training data, initial weights takes scale=10
# print(w0.min(), w0.max()) = tensor(-30.2107) tensor(32.5220)
# print(w0.grad.min(), w0.grad.max()) :tensor(-8.9067e-05) tensor(0.0002)

# ========================================================================================
## Part 2: Linear classifier with Stochastic Gradient Descent
train_sample_index = bmo.stratified_splits_sample(train_y, 10)
train_sample_index
sample_0 = train_sample_index[0]

# in pytorch, use DataLoader to do the random shuffling and batch splitting for you
gc.collect()
torch.cuda.empty_cache()
dset = list(zip(train_x, train_y))
valid_dset = list(zip(valid_x, valid_y))
dl = DataLoader(dset, batch_size=256, shuffle=True)
valid_dl = DataLoader(valid_dset, batch_size=256, shuffle=True)
xb, yb = first(dl)
w1 = bmo.init_params((xb.shape[1], 1))
b1 = bmo.init_params(1)
print(bmo.validate_one_epoch(bmo.linear1, valid_dl, w1, b1))
# starting accuracy: 0.5104
learning_rate = 10
for i in range(20):
    bmo.train_one_epoch_by_batch(bmo.linear1, bmo.mnist_loss, epoch=i,
                                 learning_rate=learning_rate, weights=w1, bias=b1 ,
                                 dl=dl)
    print(bmo.validate_one_epoch(bmo.linear1, valid_dl, w1, b1))
# Final accuracy: 0.9838 good enough

# ========================================================================================
## Part 3. Create an Optimizer
# 3.1 use nn.linear to combine init_params and linear1 together
train_x.shape[1]
linear_model = nn.Linear(train_x.shape[1], 1)
type(linear_model) # <class 'torch.nn.modules.linear.Linear'>
# what does linear_model do?
test_result = linear_model(train_x) # it's doing train_x@weights + bias
test_result.shape # torch.Size([12396, 1])
# what's in the paramaters
w,b = linear_model.parameters()
w.mean(), w.std() # (tensor(-0.0010, grad_fn=<MeanBackward0>), tensor(0.0200, grad_fn=<StdBackward0>))
# we can directly call the validate_epoch,
# here the `linear_model` contains the (1) prediction formula (2)params
print(bmo.validate_epoch(linear_model, valid_dl)) # 0.3853

# 3.2 call the optimizer class
opt = bmo.BasicOptimSGD(params=linear_model.parameters(), lr=10)
type(opt) # <class 'utils.learner_and_optimizer.BasicOptim'>

# 3.3 simplified the train epoch and train model parts
# the following 2 functions are a simplified version of my own `train_one_epoch` function
def train_epoch(model):
    for xb, yb in dl:
        bmo.calc_grad(model, xb, yb)
        opt.step() # an in-place updates to the params of the opt object
        opt.zero_grad()

def train_model(model, epoch):
    for i in range(epoch):
        train_epoch(model)
        print(bmo.validate_epoch(model, valid_dl))

train_model(linear_model, epoch=20) # last accuracy: 0.9848

# ========================================================================================
# Part 4. Here comes my Learner
# It's interesting to see that a Learner,at the time of definition, contains
# (1) the train and validation data wrapped in DataLoaders (s)
# (2) prediction model containing the activations and the parameters;
# (3) optimizer;
# (4) the loss function;
# (5) the evaluation metrics of the validation set
# The Learner is very compact. We can only use 3 lines of code to complete the training.
dls = DataLoaders(dl, valid_dl) # "s" plural

# 4.1 with fastai built-in SGD optimizer
lr_learn = Learner(dls, nn.Linear(28*28,1), opt_func=SGD,
                loss_func=bmo.mnist_loss, metrics=bmo.batch_accuracy)
epoch = 10
lr = 1
lr_learn.fit(epoch, lr=lr)
bmo.plot_epoch_stats(plt,  epoch, lr, "Simple Linear Classifier", lr_learn).savefig("training_results_linearReg.png", dpi=300, bbox_inches="tight")
'''
epoch     train_loss  valid_loss  batch_accuracy  time    
0         0.058305    0.041626    0.969578        00:00     
...  
9         0.020154    0.024984    0.979882        00:00     
'''
# 4.2 with my own optimizer + modification
# cannot call the BasicOptimSGD directly, it's incompatible with fastai's Learner module
# it causes error "AttributeError: 'BasicOptim' object has no attribute 'state'"
# so I need to wrap BasicOptimSGD into the fastiai's Optimizer
lr_learn = Learner(dls, nn.Linear(28*28,1),
                opt_func=bmo.BasicOptimSGDInherit,
                loss_func=bmo.mnist_loss, metrics=bmo.batch_accuracy)
epoch = 10
lr = 1
lr_learn.fit(epoch, lr=lr)
bmo.plot_epoch_stats(plt,  epoch, lr, "Simple Linear Classifier (BMO Optimizer)", lr_learn).savefig("training_results_linearReg_bmo_opt.png", dpi=300, bbox_inches="tight")
'''
epoch     train_loss  valid_loss  batch_accuracy  time    
0         0.058504    0.041645    0.970069        00:00     
...
9         0.020160    0.024613    0.980373        00:00     
'''

# ========================================================================================
## Part 5: A simple Neural Network with Linear and Relu layers
gc.collect()
torch.cuda.empty_cache()

simple_net = nn.Sequential(nn.Linear(28*28,30),
                           nn.ReLU(),
                           nn.Linear(30,1))
nnlearn = Learner(dls, simple_net, opt_func=SGD,
                  loss_func=bmo.mnist_loss, metrics=bmo.batch_accuracy)

# fit epoch = 40, lr = 1
epoch = 40
lr = 1
nnlearn.fit(epoch, lr)
bmo.plot_epoch_stats(plt,  epoch, lr, "Simple Neural Net", nnlearn).savefig("training_results.png", dpi=300, bbox_inches="tight")
'''
epoch     train_loss  valid_loss  batch_accuracy  time    
0         0.054991    0.030774    0.974485        00:00     
... 
39        0.007313    0.015832    0.984789        00:00     
'''
# fit epoch = 40, lr = 1
epoch = 40
lr = 1
nnlearn.fit(epoch, lr)
# Extract recorder values
bmo.plot_epoch_stats(plt,  epoch, lr, "Simple Neural Net", nnlearn).savefig("training_results2.png", dpi=300, bbox_inches="tight")
'''
epoch     train_loss  valid_loss  batch_accuracy  time    
0         0.007097    0.016058    0.985280        00:00     
...  
19        0.006910    0.016168    0.983808        00:00     
'''
