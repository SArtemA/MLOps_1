import pandas as pd
from catboost.datasets import titanic

train_df, test_df = titanic()
train_df.to_csv("train_titanic.csv")
test_df.to_csv("test_titanic.csv")
