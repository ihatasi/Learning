# これは何
ACAIと比較するためのVanilla AEです．
ACAIからCriticを抜き取ったプログラムになっています．  


ネットワークはFull Convolutionの`mnist_conv.py`とLatent Spaceが全結合の`mnist_fl.py`があり，
オプションから選択可能です(`-n {conv, fl}`)．
### オプション
`"--batchsize", "-b", type=int, default=64`
`"--epoch", "-e", type=int, default=100`
`"--gpu", "-g", type=int, default=0`
`"--snapshot", "-s", type=int, default=10`
`"--n_dimz", "-z", type=int, default=16`
`"--dataset", "-d", type=str, default='mnist'`
`"--network", "-n", type=str, default='conv'`
