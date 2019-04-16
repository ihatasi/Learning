import chainer
import chainer.functions as F
import chainer.links as L

class AE(chainer.Chain):

    def __init__(self, n_dimz):
        self.n_dimz = n_dimz
        super(AE, self).__init__()

        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 16, 5)
            self.deconv2 = L.Deconvolution2D(None, 3, 5)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.deconv2(h)
        return F.sigmoid(h)
