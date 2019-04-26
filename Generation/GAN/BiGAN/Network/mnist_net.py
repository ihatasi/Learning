#https://github.com/chainer/chainer/tree/v5/examples/dcgan
#!/usr/bin/env python

import numpy
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import backend

class Generator(chainer.Chain):
    #Input:z Output:x
    def __init__(self, n_hidden, bottom_width=7, ch=512, wscale=0.02):
        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width

        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(self.n_hidden, 1024, initialW=w)
            self.l1 = L.Linear(1024, bottom_width*bottom_width*128, initialW=w)
            self.dc1 = L.Deconvolution2D(in_channels=128, out_channels=64,
                ksize=4, stride=2, pad=1, initialW=w)
            self.dc2 = L.Deconvolution2D(None, 1, 4, 2, 1, initialW=w)
            self.bnl0 = L.BatchNormalization(1024)
            self.bnl1 = L.BatchNormalization(bottom_width * bottom_width * 128)
            self.bndc1 = L.BatchNormalization(64)

    def make_hidden(self, batchsize):
        dtype = chainer.get_dtype()
        return numpy.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1))\
            .astype(dtype)

    def forward(self, z):
        h = F.relu(self.bnl0(self.l0(z)))
        h = F.relu(self.bnl1(self.l1(h)))
        h = F.reshape(h,
                      (-1, 128, self.bottom_width, self.bottom_width))
        h = F.relu(self.bndc1(self.dc1(h)))
        x = F.sigmoid(self.dc2(h))
        return x


class Discriminator(chainer.Chain):
    #Input:(x,z) Output:(0,1)
    def __init__(self, bottom_width=3, ch=512, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(Discriminator, self).__init__()
        with self.init_scope():
            #x layer
            self.cx_0 = L.Convolution2D(in_channels=1, out_channels=64,
                ksize=4, stride=2, pad=1, initialW=w)
            self.cx_1 = L.Convolution2D(None, 64, 4, 2, 1, initialW=w)
            #z layer
            self.cz_0 = L.Linear(None, 512, initialW=w)
            self.bnx_1 = L.BatchNormalization(64, use_gamma=False)
            #y layer
            self.y_1 = L.Linear(None, 1024, initialW=w)
            self.y_2 = L.Linear(None, 1, initialW=w)

    def forward(self, x, z):
        hx = F.leaky_relu(self.cx_0(x)) #[1, 28, 28]->[64, 10, 10]
        hx = F.leaky_relu(self.bnx_1(self.cx_1(hx))) #[64, 14, 14]->[128, 6, 6]
        hx = F.reshape(hx, (hx.data.shape[0], -1))
        hz = F.leaky_relu(self.cz_0(z)) #[100]->[512]
        h = F.concat([hx, hz], axis=1)
        h1 = F.leaky_relu(self.y_1(h))
        h2 = self.y_2(h1)
        return h1, h2

class Encoder(chainer.Chain):
    #Input:x, Output:z
    def __init__(self, n_hidden, ch=512, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        self.n_hidden = n_hidden
        super(Encoder, self).__init__()
        with self.init_scope():
            self.c0_0 = L.Convolution2D(in_channels=1, out_channels=32,
                ksize=3, stride=1, pad=1, initialW=w)
            self.c1_0 = L.Convolution2D(32, 64, 3, 2, 1, initialW=w)
            self.c2_0 = L.Convolution2D(64, 128, 3, 2, 1, initialW=w)
            self.l4 = L.Linear(None, self.n_hidden, initialW=w)
            self.bn0_0 = L.BatchNormalization(32, use_gamma=False)
            self.bn1_0 = L.BatchNormalization(64, use_gamma=False)
            self.bn2_0 = L.BatchNormalization(128, use_gamma=False)

    def forward(self, x):
        h = F.leaky_relu(self.c0_0(x))
        h = F.leaky_relu(self.bn1_0(self.c1_0(h)))
        h = F.leaky_relu(self.bn2_0(self.c2_0(h)))
        h = self.l4(h)
        return h
