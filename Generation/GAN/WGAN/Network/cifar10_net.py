#https://github.com/chainer/chainer/tree/v5/examples/dcgan
#!/usr/bin/env python

import numpy
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import backend

class Generator(chainer.Chain):

    def __init__(self, n_hidden, bottom_width=4, ch=512, wscale=0.001):
        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width
        self.use_bn= True

        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(self.n_hidden, bottom_width * bottom_width * ch,
                               initialW=w)
            #4x4
            self.dc1 = L.Deconvolution2D(ch, ch // 2, 4, 2, 1, initialW=w)
            #8x8
            self.dc2 = L.Deconvolution2D(ch // 2, ch // 4, 4, 2, 1, initialW=w)
            #16x16
            self.dc3 = L.Deconvolution2D(ch // 4, ch // 8, 4, 2, 1, initialW=w)
            #32x32
            self.dc4 = L.Deconvolution2D(ch // 8, 3, 3, 1, 1, initialW=w)
            #32x32
            if self.use_bn:
                self.bn0 = L.BatchNormalization(bottom_width * bottom_width * ch)
                self.bn1 = L.BatchNormalization(ch // 2)
                self.bn2 = L.BatchNormalization(ch // 4)
                self.bn3 = L.BatchNormalization(ch // 8)

    def make_hidden(self, batchsize):
        dtype = chainer.get_dtype()
        return numpy.random.normal(size=(batchsize, 1, 4, 4))\
            .astype(dtype)

    def __call__(self, z):
        if not self.use_bn:
            h = F.reshape(F.relu(self.l0(z)),
                          (len(z), self.ch, self.bottom_width, self.bottom_width))
            h = F.relu(self.dc1(h))
            h = F.relu(self.dc2(h))
            h = F.relu(self.dc3(h))
            x = self.dc4(h)
        else:
            h = F.reshape(F.relu(self.bn0(self.l0(z))),
                          (len(z), self.ch, self.bottom_width, self.bottom_width))
            h = F.relu(self.bn1(self.dc1(h)))
            h = F.relu(self.bn2(self.dc2(h)))
            h = F.relu(self.bn3(self.dc3(h)))
            x = self.dc4(h)
        return F.sigmoid(x)

class Critic(chainer.Chain):

    def __init__(self, bottom_width=4, ch=512, wscale=0.001, output_dim=32*32):
        w = chainer.initializers.Normal(wscale)
        super(Critic, self).__init__()
        with self.init_scope():
            #32x32
            #self.c0 = L.Convolution2D(3, ch // 8, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(3, ch // 4, 4, 2, 1, initialW=w)
            #16x16
            #self.c1_0 = L.Convolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w)
            self.c2 = L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w)
            #8x8
            #self.c2_0 = L.Convolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w)
            self.c3 = L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w)
            #4x4
            #self.c3_0 = L.Convolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w)
            #self.l4 = L.Linear(bottom_width * bottom_width * ch, output_dim, initialW=w)

    def __call__(self, x):
        #h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.c1(x))
        #h = F.leaky_relu(self.c1_0(h))
        h = F.leaky_relu(self.c2(h))
        #h = F.leaky_relu(self.c2_0(h))
        h = self.c3(h)
        #h = F.leaky_relu(self.c3(h))
        #h = self.c3_0(h)
        #h = self.l4(h)
        return h
