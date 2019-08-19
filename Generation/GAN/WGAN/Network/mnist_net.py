import sys
import os

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import numpy as np

class DCGANGenerator(chainer.Chain):
    def __init__(self, n_hidden=128, ch=128, bottom_size=4):
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_size = bottom_size
        super(DCGANGenerator, self).__init__()
        with self.init_scope():
            #self.fc1=L.Linear(None, 500)
            #self.fc2=L.Linear(None, 28 * 28)
            self.l0=L.Linear(None, self.bottom_size*self.bottom_size*self.ch)
            self.dc0=L.Deconvolution2D(None, ch//1, 4, 2, 1)
            self.dc1=L.Deconvolution2D(None, ch//2, 4, 2, 1)
            self.dc2=L.Deconvolution2D(None, ch//4, 4, 2, 1)
            self.dc3=L.Deconvolution2D(None, 1, 3, 1, 3)

    def __call__(self, z, test=False):
        #h = F.relu(self.fc1(z))
        #h = F.reshape(F.sigmoid(self.fc2(h)), (-1, 1, 28, 28))
        h = F.reshape(F.relu(self.l0(z)), (len(z), self.ch, self.bottom_size, self.bottom_size))
        h = F.relu(self.dc0(h))
        h = F.relu(self.dc1(h))
        h = F.relu(self.dc2(h))
        h = F.sigmoid(self.dc3(h))
        return h
    def make_hidden(self, batchsize):
        return np.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)) \
            .astype(np.float32)


class WGANDiscriminator(chainer.Chain):
    def __init__(self, n_hidden=128, ch=128, bottom_size=4):
        super(WGANDiscriminator, self).__init__()
        with self.init_scope():
            #self.fc1=L.Linear(None, 500)
            #self.fc2=L.Linear(None, 28 * 28)
            self.c0=L.Convolution2D(None, 1, 3, 1, 3)
            self.c1=L.Convolution2D(None, ch//4, 4, 2, 1)
            self.c2=L.Convolution2D(None, ch//2, 4, 2, 1)
            self.c3=L.Convolution2D(None, ch//1, 4, 2, 1)

    def __call__(self, x, test=False):
        #h = F.relu(self.fc1(x))
        #h = self.fc2(h)
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.c1(h))
        h = F.leaky_relu(self.c2(h))
        h = self.c3(h)
        return h