# これは何？
Chainerで書いたディープのコード(公開用)です．
基本的に自分用に書いて，公開できるコード（他人の論文のChainer実装）が置いてあります．<br>
## 使い方
Nvidia-dockerを利用しています．
`nvidia-docker build -t chainer5 -f Dockerfile .`
でビルドした後に，各ディレクトリに置いてある`run.sh`を実行して環境に入ってください．<br>
### 現在実装されているもの一覧
#### AE
 - Vanilla AE
 - Convolution AE
 - ACAI
#### GAN
 - DCGAN
 - AnoGAN
 - ADGAN
 - BiGAN
 - WGAN
 - WGAN-gp
 - f-AnoGAN

