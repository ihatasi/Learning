import chainer
import chainer.functions as F
import chainer.links as L
import random
import cupy

class Encoder(chainer.Chain):
    def __init__(self, n_dimz):
        self.n_dimz = n_dimz
        super(Encoder, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 16, 3, 1, 1)
            self.conv2 = L.Convolution2D(None, 32, 4, 2, 1)
            self.conv3 = L.Convolution2D(None, 64, 4, 2, 1)
            self.conv_z = L.Convolution2D(None, self.n_dimz, 4, 2, 1)
            self.bn1 = L.BatchNormalization(16)
            self.bn2 = L.BatchNormalization(32)
            self.bn3 = L.BatchNormalization(64)
    def __call__(self, x):
        h = self.bn1(F.relu(self.conv1(x)))
        h = self.bn2(F.relu(self.conv2(h)))
        h = self.bn3(F.relu(self.conv3(h)))
        h = self.conv_z(h)
        return h

class Decoder(chainer.Chain):
    def __init__(self, n_dimz):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.z_deconv = L.Deconvolution2D(None, 64, 4, 2, 1)
            self.deconv1 = L.Deconvolution2D(None, 32, 4, 2, 1)
            self.deconv2 = L.Deconvolution2D(None, 16, 4, 2, 1)
            self.deconv3 = L.Deconvolution2D(None, 1, 3, 1, 1)
            self.bn1 = L.BatchNormalization(64)
            self.bn2 = L.BatchNormalization(32)
            self.bn3 = L.BatchNormalization(16)
    def __call__(self, x):
        h = self.bn1(F.relu(self.z_deconv(x)))
        h = self.bn2(F.relu(self.deconv1(h)))
        h = self.bn3(F.relu(self.deconv2(h)))
        h = self.deconv3(h)
        return F.sigmoid(h)


class Critic(chainer.Chain):
    def __init__(self):
        super(Critic, self).__init__()
        with self.init_scope():
            self.dis0 = L.Convolution2D(None, 16, 3, 1, 1)
            self.dis1 = L.Convolution2D(None, 16, 4, 2, 1)
            self.dis2 = L.Convolution2D(None, 32, 4, 2, 1)
            self.dis3 = L.Convolution2D(None, 64, 4, 2, 1)
            self.dis4 = L.Linear(None, 1)
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
