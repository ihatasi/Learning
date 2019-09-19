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
            self.conv1 = L.Convolution2D(None, 16, 3, 1, 1)
            self.conv2 = L.Convolution2D(None, 32, 4, 2, 1)
            self.conv3 = L.Convolution2D(None, 64, 4, 2, 1)
            self.conv_z = L.Convolution2D(None, self.n_dimz, 4, 2, 1)
            self.z_deconv = L.Deconvolution2D(None, 64, 4, 2, 1)
            self.deconv1 = L.Deconvolution2D(None, 32, 4, 2, 1)
            self.deconv2 = L.Deconvolution2D(None, 16, 4, 2, 1)
            self.deconv3 = L.Deconvolution2D(None, 1, 3, 1, 1)

    def __call__(self, x1, x2, train=True):
        if train:
            batchsize = x1.shape[0]
            xp = cupy
            alpha = chainer.Variable(xp.random.rand(batchsize, dtype=xp.float32))
            alpha = 0.5 - F.absolute(0.5 - alpha)
            alpha = alpha.reshape(batchsize, 1, 1 ,1)
            h1 = F.relu(self.conv1(x1))
            h2 = F.relu(self.conv1(x2))
            h1 = F.relu(self.conv2(h1))
            h2 = F.relu(self.conv2(h2))
            h1 = F.relu(self.conv3(h1))
            h2 = F.relu(self.conv3(h2))
            h1 = self.conv_z(h1)
            h2 = self.conv_z(h2)
            c =  alpha*h1+(1.0-alpha)*h2
            y1 = self.z_deconv(h1)
            y2 = self.z_deconv(h2)
            yc = self.z_deconv(c)
            y1 = F.relu(self.deconv1(y1))
            y2 = F.relu(self.deconv1(y2))
            yc = F.relu(self.deconv1(yc))
            y1 = F.relu(self.deconv2(y1))
            y2 = F.relu(self.deconv2(y2))
            yc = F.relu(self.deconv2(yc))
            y1 = self.deconv3(y1)
            y2 = self.deconv3(y2)
            yc = self.deconv3(yc)
            return F.sigmoid(y1), F.sigmoid(y2), F.sigmoid(yc), alpha, h1, h2
        else:
            y = F.relu(self.z_deconv(x1))
            y = F.relu(self.deconv1(y))
            y = F.relu(self.deconv2(y))
            y = self.deconv3(y)
            return F.sigmoid(y)


class Critic(chainer.Chain):
    def __init__(self):
        super(Critic, self).__init__()
        with self.init_scope():
            self.dis0 = L.Convolution2D(None, 16, 3, 1, 1)
            self.dis1 = L.Convolution2D(None, 16, 4, 2, 1)
            self.dis2 = L.Convolution2D(None, 32, 4, 2, 1)
            self.dis3 = L.Convolution2D(None, 64, 4, 2, 1)
            self.dis4 = L.Linear(None, 1)
    def __call__(self, x):
        h = F.relu(self.dis0(x))
        h = F.relu(self.dis1(h))
        h = F.relu(self.dis2(h))
        h = F.relu(self.dis3(h))
        h = self.dis4(h)
        return F.sigmoid(h)/2
