#https://github.com/pfnet-research/chainer-gan-lib/blob/master/common/net.py
import sys
import os

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import numpy as np

class AE(chainer.Chain):
    def __init__(self, n_hidden=128, ch=128, bottom_size=4):
        self.ch = ch
        super(AE, self).__init__()
        with self.init_scope():
            self.c0=L.Convolution2D(None, 1, 3, 1, 3)
            self.c1=L.Convolution2D(None, ch//4, 4, 2, 1)
            self.c2=L.Convolution2D(None, ch//2, 4, 2, 1)
            self.c3=L.Convolution2D(None, ch//1, 4, 2, 1)
            self.l1=L.Linear(None, ch)
    def __call__(self, x):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.c1(h))
        h = F.leaky_relu(self.c2(h))
        h = self.c3(h)
        h = self.l1(h)
        return h.reshape(-1, self.ch, 1, 1)