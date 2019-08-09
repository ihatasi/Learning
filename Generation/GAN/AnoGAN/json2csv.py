import pandas as pd


df0 = pd.read_json("result/mnist/log")


df0.to_csv("result/mnist/log.csv")
