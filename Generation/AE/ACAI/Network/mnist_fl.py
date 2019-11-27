import chainer
import chainer.functions as F
import chainer.links as L
import random
import cupy
import numpy as np

class Encoder(chainer.Chain):
    def __init__(self, n_dimz):
        self.n_dimz = n_dimz
        super(Encoder, self).__init__()
        with self.init_scope():
            w = chainer.initializers.Normal(0.02)
            self.conv1 = L.Convolution2D(None, 16, 3, 1, 1, initialW=w)
            self.conv2 = L.Convolution2D(None, 32, 4, 2, 1, initialW=w)
            self.conv3 = L.Convolution2D(None, 64, 4, 2, 1, initialW=w)
            self.conv_z = L.Linear(None, self.n_dimz, initialW=w)
            self.bn1 = L.BatchNormalization(16)
            self.bn2 = L.BatchNormalization(32)
            self.bn3 = L.BatchNormalization(64)
    def __call__(self, x):
        h = self.bn1(F.leaky_relu(self.conv1(x)))
        h = self.bn2(F.leaky_relu(self.conv2(h)))
        h = self.bn3(F.leaky_relu(self.conv3(h)))
        h = self.conv_z(h)
        return h

class Decoder(chainer.Chain):
    def __init__(self, n_dimz):
        self.n_hidden = n_dimz
        super(Decoder, self).__init__()
        with self.init_scope():
            w = chainer.initializers.Normal(0.02)
            self.z_deconv = L.Linear(None, 64*8*8, initialW=w)
            self.deconv1 = L.Deconvolution2D(None, 32, 4, 2, 1, initialW=w)
            self.deconv2 = L.Deconvolution2D(None, 16, 4, 2, 1, initialW=w)
            self.deconv3 = L.Deconvolution2D(None, 1, 3, 1, 1, initialW=w)
            self.bn1 = L.BatchNormalization(64)
            self.bn2 = L.BatchNormalization(32)
            self.bn3 = L.BatchNormalization(16)
    def __call__(self, x):
        h = self.z_deconv(x).reshape(-1, 64, 8, 8)
        h = self.bn2(F.leaky_relu(self.deconv1(h)))
        h = self.bn3(F.leaky_relu(self.deconv2(h)))
        h = self.deconv3(h)
        return F.sigmoid(h)



class Critic(chainer.Chain):
    def __init__(self):
        super(Critic, self).__init__()
        with self.init_scope():
            w = chainer.initializers.Normal(0.02)
            self.dis0 = L.Convolution2D(None, 16, 3, 1, 1, initialW=w)
            self.dis1 = L.Convolution2D(None, 16, 4, 2, 1, initialW=w)
            self.dis2 = L.Convolution2D(None, 32, 4, 2, 1, initialW=w)
            self.dis3 = L.Convolution2D(None, 64, 4, 2, 1, initialW=w)
            self.dis4 = L.Linear(None, 1, initialW=w)
            self.bn0= L.BatchNormalization(16)
            self.bn1 = L.BatchNormalization(16)
            self.bn2 = L.BatchNormalization(32)
            self.bn3 = L.BatchNormalization(64)
    def __call__(self, x):
        h = self.bn0(F.relu(self.dis0(x)))
        h = self.bn1(F.relu(self.dis1(h)))
        h = self.bn2(F.relu(self.dis2(h)))
        h = self.bn3(F.relu(self.dis3(h)))
        h = self.dis4(h)
        return 0.5 - F.absolute(0.5 - F.sigmoid(h))

