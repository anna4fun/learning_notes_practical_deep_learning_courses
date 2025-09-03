import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fastai.vision.all import *
from fastcore.parallel import *
import os
from PIL import Image
import MNIST_Handwritten_Digits_Classifier.utils.learner_and_optimizer as bmo
import gc
import importlib

importlib.reload(bmo)

# import MNIST dataset
# URLs is an object that stores all the URLs of datasets, in this tutorial we are going to use the  MNIST_SAMPLE and MINIST_TINY
# let's define lists of path to images which will be used for data loading
#path = untar_data(URLs.MNIST_SAMPLE) # Path('/Users/lcjh/.fastai/data/mnist_sample')
path = Path('/Users/lcjh/.fastai/data/mnist_sample')
sevens = (path/'train'/'7').ls().sorted()
threes = (path/'train'/'3').ls().sorted()
valid_sevens = (path/'valid'/'7').ls().sorted()
valid_threes = (path/'valid'/'3').ls().sorted()

# How many images are there?
print(len(threes)) # 6131
print(len(sevens)) # 6265

# Sanity check 1: randomly pick one images from each set to eye-ball the validity of the image
m3t = tensor(Image.open(threes[np.random.randint(len(threes))]))
# print_image_as_pixel(m3t)
show_image(m3t)
m7t = tensor(Image.open(sevens[np.random.randint(len(sevens))]))
# print_image_as_pixel(m7t)
show_image(m7t)

# Sanity check 2: all images of the same size? pass the file path to the get_image_files() func
train_files = get_image_files(path/'train')
train_file_size = parallel(bmo.file_size, train_files, n_workers=10)
file_sizes = pd.Series(train_file_size).value_counts() # (28, 28)    12396

# Loading the training and validation images into DataSets
# Use stack to turn nested list into a rank-3 tensor. Because the nested list still not easy to iterate and average each images.
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
valid_y = tensor([1]*stacked_three_valid.shape[0]+[0]*stacked_seven_valid.shape[0]).unsqueeze(1)
valid_y[stacked_three_valid.shape[0]]

model_data = bmo.ModelData(train_x, train_y, valid_x, valid_y)
# test the validation accuracy function
# making sure the dimensions are all correct
w0=bmo.init_params((train_x.shape[1],1))
b0=bmo.init_params(1)
w0.shape, b0.shape # (torch.Size([784, 1]), torch.Size([1]))
pred = bmo.linear1(model_data.valid_x, w0, b0)
pred.shape
acc = bmo.batch_accuracy(pred, model_data.valid_y)
acc, acc.shape
loss = bmo.mnist_loss(pred, model_data.valid_y)
loss.shape, loss.item() # (torch.Size([]), 0.5325248837471008)
# In PyTorch, torch.Size([]) means a 0-dim (scalar) tensor.
# Many losses default to a reduction of "mean" (or "sum"), which collapses the batch and returns a single scalar value.
# tensor.item() returns the floats saved in the tensor, which could then be rounded by round() function
training_epoch_results = []

for learning_rate in [100, 150, 200]:
    gc.collect()
    torch.cuda.empty_cache()
    weights = bmo.init_params((train_x.shape[1], 1), 10)
    bias = bmo.init_params(1)
    for i in range(20):
        results = bmo.train_one_epoch(bmo.linear1, bmo.mnist_loss, model_data, epoch=i,
                                      learning_rate=learning_rate, weights=weights, bias=bias)
        training_epoch_results.append(results)

results_df = pd.DataFrame(training_epoch_results)
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

# saving the results for future reference
cwd = os.getcwd()
output_dir = cwd + '/MNIST_Handwritten_Digits_Classifier/results/digit_classifier_results.csv'
results_df.to_csv(output_dir, index=False)

# key take aways:
# 1. configure the dimensions of each tensors objects carefully
# 2. check the scale of weights and gradients, if the scale differs too much, select a big learning_rate to step
# 3. last but not least, save all the outputs with a history of all the variations that I tried for more efficient experiments




