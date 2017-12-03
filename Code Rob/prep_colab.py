import pandas as pd
import numpy as np
import json
import _pickle as pickle

DIR = "data/"
filename = "train.csv"
file = DIR + filename

df = pd.read_csv(file, index_col=2)

users = list(df.index.values)
num_users = len(users)

R = {}
i = 0
for user in users[:100]:
    i += 1
    print("Checked: " + str(i) + " out of " + str(num_users))
    records = df.loc[[user]]
    rated = records.loc[:, "Track"]
    R[user] = list(rated)

with open('data.p', 'wb') as fp:
    pickle.dump(R, fp)