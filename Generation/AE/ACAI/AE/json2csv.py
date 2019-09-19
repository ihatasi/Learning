import pandas as pd


df0 = pd.read_json("result/log")


df0.to_csv("result/log.csv")