import chainer
import chainer.functions as F
import chainer.links as L
import random
import cupy

class AE(chainer.Chain):
    def __init__(self, n_dimz, batchsize):
        self.n_dimz = n_dimz
        self.batchsize = batchsize
        super(AE, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 16, 4, 2, 1)
            self.conv2 = L.Convolution2D(None, 32, 4, 2, 1)
            self.conv3 = L.Convolution2D(None, 64, 4, 2, 1)
            self.conv_z = L.Linear(None, self.n_dimz)
            self.z_deconv = L.Linear(None, 64*4*4)
            self.deconv1 = L.Deconvolution2D(None, 32, 4, 2, 1)
            self.deconv2 = L.Deconvolution2D(None, 16, 4, 2, 1)
            self.deconv3 = L.Deconvolution2D(None, 1, 4, 2, 1)

    def __call__(self, x1, x2, train=True):
        if train:
            h1 = F.relu(self.conv1(x1))
            h2 = F.relu(self.conv1(x2))
            h1 = F.relu(self.conv2(h1))
            h2 = F.relu(self.conv2(h2))
            h1 = F.relu(self.conv3(h1))
            h2 = F.relu(self.conv3(h2))
            h1 = self.conv_z(h1)
            h2 = self.conv_z(h2)
            y1 = self.z_deconv(h1)
            y2 = self.z_deconv(h2)
            y1 = F.relu(self.deconv1(y1.reshape(-1, 64, 4, 4)))
            y2 = F.relu(self.deconv1(y2.reshape(-1, 64, 4, 4)))
            y1 = F.relu(self.deconv2(y1))
            y2 = F.relu(self.deconv2(y2))
            y1 = self.deconv3(y1)
            y2 = self.deconv3(y2)
            return F.sigmoid(y1), F.sigmoid(y2), h1, h2
        else:
            y = F.relu(self.z_deconv(x1)).reshape(-1, 64, 4, 4)
            y = F.relu(self.deconv1(y))
            y = F.relu(self.deconv2(y))
            y = self.deconv3(y)
            return F.sigmoid(y)
