from fastai import *
from fastai.vision.all import * #this one import torch
import gc
# import pynvml # only on linux and windows machine
import torch
print("MPS built:", getattr(torch.backends, "mps", None) and torch.backends.mps.is_built())
print("MPS available:", getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
