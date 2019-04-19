# これは何？
AnoGANの追試をしたものです．<br>
AnoGANについては（https://ihatasi.hatenablog.com/entry/2019/04/20/030339）に書いてあるので，ここではプログラムの動かし方について書きます．<br>
ここではNvidia-dockerを使っています．環境がLearningのDockerファイルに書かれているので，それを実行するか同じ環境を整えてください．
# 動かし方
`python3 Train.py`でDCGANの学習済みモデルを読み込み学習が始まります．<br>
**このフォルダ内にあるDCGANをまだ動かしていない場合はそちらを先に動かしてください．**
ディレクトリ移動して`python3 Train.py`で動きます．<br>
学習中はデフォルトでは10Epochに1回入力画像と生成画像を並べた画像が./result/previewに出力されます．<br>
オプションの引数が設定されているので，学習回数などを変えたいときはTrain.py内のAurgumentParserを見て変えてください．<br>
`json2csv.py`はダンプされたログファイルをCSV形式に変えるプログラムです．<br>
`Predictor.py`は学習モデルを読み込んで再現するためのプログラムです．<br>
