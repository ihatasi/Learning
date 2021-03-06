#!/usr/bin/python3

import chainer
import chainer.functions as F
from chainer import Variable


class AnoGANUpdater(chainer.training.updaters.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis, self.ser = kwargs.pop('models')
        self.z_noise = kwargs.pop('z_noise')
        super(AnoGANUpdater, self).__init__(*args, **kwargs)

    def loss_dis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        #real label 1
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        #fake label 0
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def loss_gen(self, gen, y_fake):
        batchsize = len(y_fake)
        #fake label 1
        loss = F.sum(F.softplus(-y_fake)) / batchsize
        chainer.report({'loss': loss}, gen)
        return loss

    def loss_Ano(self, ser, x_real, x_fake, y_fake, y_real):
        lam = 0.1
        batchsize = len(y_fake)
        #Residual Loss
        L1 = F.sum(F.absolute_error(x_real, x_fake)) / batchsize
        #Discrimination Loss
        L2 = F.sum(F.absolute_error(y_real, y_fake)) / batchsize
        loss = (1-lam)*L1 + lam*L2
        chainer.report({'loss': loss}, ser)
        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')
        ser_optimizer = self.get_optimizer('ser')

        batch = self.get_iterator('main').next()
        x_real = Variable(self.converter(batch, self.device))
        xp = chainer.backend.get_array_module(x_real.data)

        gen, dis, ser= self.gen, self.dis, self.ser
        batchsize = len(batch)
        #gen.disable_update() #重み固定処理（Updateしてないので入れなくてよい）
        #dis.disable_update() #重み固定処理（Updateしてないので入れなくてよい）

        z_noise = ser(self.z_noise)

        """
        DCGANのTrain時には1バッチに複数枚画像が入っているが，
        AnoGANのTrain時は1バッチ1枚しかない．
        同じネットワークを使うためにも以下の処理をしないといけない．
        """
        with chainer.using_config('train', False):#for BacthNormalization
            y_real = dis(x_real)
            x_fake = gen(z_noise)
            y_fake = dis(x_fake)
        self.loss_dis(dis, y_fake, y_real)
        self.loss_gen(gen, y_fake)
        """update optimizers"""
        ser_optimizer.update(self.loss_Ano, ser, x_real, x_fake, y_fake, y_real)
