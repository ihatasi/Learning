# これは何
(https://arxiv.org/abs/1807.07543)の論文内のACAIのChainerでのテスト実装です．
## 使い方
1. Train.pyを実行
2. Predictor.pyを実行

Predicotor.pyを実行するとpictフォルダ内の画像が生成されます．
必要があれば自分の使いやすいように書き換えたり，オプションをつけてください．
ネットワークはFull Convolutionの`mnist_conv.py`とLatent Spaceが全結合の`mnist_fl.py`があり，
オプションから選択可能です(`-n {conv, fl}`)．
### オプション
`"--batchsize", "-b", type=int, default=64`<br>
`"--epoch", "-e", type=int, default=100`<br>
`"--gpu", "-g", type=int, default=0`<br>
`"--snapshot", "-s", type=int, default=10`<br>
`"--n_dimz", "-z", type=int, default=16`<br>
`"--dataset", "-d", type=str, default='mnist'`<br>
`"--network", "-n", type=str, default='conv'`<br>

