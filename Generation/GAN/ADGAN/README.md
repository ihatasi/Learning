# これは何？
ChainerによるADGANのプログラムです．
ADGANについては（ https://ihatasi.hatenablog.com/entry/2019/05/16/032530 ） に書いてあるので，ここではプログラムの動かし方について書きます．<br>
ここではNvidia-dockerを使っています．環境が/LearningのDockerファイルに書かれているので，
それをBuildするか同じ環境を整えてください．
# 動かし方
`python3 Train.py`でDCGANの学習済みモデルを読み込み学習が始まります．
このディレクトリ内にあるDCGANをまだ動かしていない場合は./DCGANでそちらを先に動かしてください．<br>
学習中はデフォルトでは10Epochごとに入力画像と出力画像のペアが./result/previewに出力されます．<br>
`json2csv.py`はダンプされたログファイルをCSV形式に変えるプログラムです．<br>
`Predictor.py`は学習モデルを読み込んで再現するためのプログラムです．
