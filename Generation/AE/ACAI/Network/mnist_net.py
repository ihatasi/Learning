import chainer
import chainer.functions as F
import chainer.links as L
import random

class AE(chainer.Chain):

    def __init__(self, n_dimz):
        self.n_dimz = n_dimz
        self.alpha = 0
        super(AE, self).__init__()

        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 16, 3, 1, 1)
            self.z = L.Linear(None, self.n_dimz)
            self.fl = L.Linear(None, 16*28*28)
            self.deconv1 = L.Deconvolution2D(None, 1, 3, 1, 1)

    def __call__(self, x1, x2, train=True):
        if train:
            batchsize = x1.shape[0]
            xp = chainer.backend.get_array_module(x1.data)
            self.alpha = xp.random.rand(batchsize)
            self.alpha = xp.vstack((self.alpha, self.alpha)).T
            h1 = self.conv1(x1)
            h2 = self.conv1(x2)
            h1 = self.z(h1)
            h2 = self.z(h2)
            c = self.alpha*h1+(1-self.alpha)*h2
            y1 = self.fl(h1).reshape(-1, 16, 28, 28)
            y2 = self.fl(h2).reshape(-1, 16, 28, 28)
            yc = self.fl(c).reshape(-1, 16, 28, 28)
            y1 = self.deconv1(y1)
            y2 = self.deconv1(y2)
            yc = self.deconv1(yc)
            return F.sigmoid(y1), F.sigmoid(y2), F.sigmoid(yc), self.alpha, h1, h2
        else:
            x1 = x1.reshape(1, self.n_dimz)
            y = self.fl(x1).reshape(-1, 16, 28, 28)
            y = self.deconv1(y)
            return F.sigmoid(y)

class Critic(chainer.Chain):
    def __init__(self):
        super(Critic, self).__init__()
        with self.init_scope():
            self.dis1 = L.Linear(None, 1)
    def __call__(self, x):
        h = self.dis1(x)
        return h
