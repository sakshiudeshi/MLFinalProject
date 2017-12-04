import pandas as pd
import numpy as np

DATA_DIR = "data"
filename = "train.csv"
file = DATA_DIR + "/" + filename

df = pd.read_csv(file)

num_users = max(df.User)
num_tracks = max(df.Track)

ratings = np.zeros((num_users, num_tracks))

TB = np.matrix(df)

for i in range(len(TB)):
    user = TB[i, 0]
    track = TB[i, 1]
    rating = TB[i, 3]
    ratings[user - 1, track - 1] = rating

print(ratings)
print(ratings.shape)