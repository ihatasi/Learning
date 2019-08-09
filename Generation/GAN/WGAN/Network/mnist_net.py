#https://github.com/chainer/chainer/tree/v5/examples/dcgan
#!/usr/bin/env python

import numpy
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import backend

class Generator(chainer.Chain):

    def __init__(self, n_hidden, bottom_width=3, ch=512, wscale=0.02):
        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width

        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(self.n_hidden, bottom_width * bottom_width * ch,
                               initialW=w)
            self.dc1 = L.Deconvolution2D(in_channels=ch, out_channels=ch // 2,
                ksize=2, stride=2, pad=1, initialW=w)
            self.dc2 = L.Deconvolution2D(ch // 2, ch // 4, 2, 2, 1, initialW=w)
            self.dc3 = L.Deconvolution2D(ch // 4, ch // 8, 2, 2, 1, initialW=w)
            self.dc4 = L.Deconvolution2D(ch // 8, 1, 3, 3, 1, initialW=w)
            self.bn0 = L.BatchNormalization(bottom_width * bottom_width * ch)
            self.bn1 = L.BatchNormalization(ch // 2)
            self.bn2 = L.BatchNormalization(ch // 4)
            self.bn3 = L.BatchNormalization(ch // 8)

    def make_hidden(self, batchsize):
        dtype = chainer.get_dtype()
        return numpy.random.normal(size=(batchsize, 1, 7, 7))\
            .astype(dtype)

    def forward(self, z):
        h = F.relu(self.bn0(self.l0(z)))
        h = F.reshape(h,
                      (-1, self.ch, self.bottom_width, self.bottom_width))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        x = F.sigmoid(self.dc4(h))
        return x


class Critic(chainer.Chain):

    def __init__(self, bottom_width=3, ch=512, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(Critic, self).__init__()
        with self.init_scope():
            self.c0_0 = L.Convolution2D(in_channels=1, out_channels=ch // 8,
                ksize=3, stride=3, pad=1, initialW=w)
            self.c1_0 = L.Convolution2D(ch // 8, ch // 4, 2, 2, 1, initialW=w)
            self.c2_0 = L.Convolution2D(ch // 4, ch // 2, 2, 2, 1, initialW=w)
            self.c3_0 = L.Convolution2D(ch // 2, ch // 1, 2, 2, 1, initialW=w)
            self.l4 = L.Linear(None, 28*28, initialW=w)
            self.bn0_0 = L.BatchNormalization(ch // 8, use_gamma=False)
            self.bn1_0 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn2_0 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn3_0 = L.BatchNormalization(ch // 1, use_gamma=False)

    def forward(self, x):
        batchsize = x.shape[0]
        h = F.leaky_relu(self.c0_0(x))
        h = F.leaky_relu(self.bn1_0(self.c1_0(h)))
        h = F.leaky_relu(self.bn2_0(self.c2_0(h)))
        h = F.leaky_relu(self.bn3_0(self.c3_0(h)))
        h = F.sum(self.l4(h)) / batchsize
        return h
