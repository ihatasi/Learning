import chainer
import chainer.functions as F
import chainer.links as L

class AE(chainer.Chain):

    def __init__(self, n_dimz, n_out):
        self.n_dimz = n_dimz
        self.n_out = n_out
        super(AE, self).__init__()

        with self.init_scope():
            self.l1 = L.Linear(None, self.n_dimz)
            self.l2 = L.Linear(None, n_out)

    def __call__(self, x):
        h = self.l1(x)
        h = self.l2(h)
        return F.sigmoid(h)
