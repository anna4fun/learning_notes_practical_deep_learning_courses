from unittest import TestCase
from fastai import *
from fastai.vision.all import *


class Test(TestCase):
    def test_class_accuracy(self):
        test_tensor = tensor([0, 0, 1])
        valid_tensor = tensor([1, 1, 1])
        if class_accuracy(test_tensor, valid_tensor) == tensor(0.3):
            self.assertTrue(True)
        else:
            self.fail()
