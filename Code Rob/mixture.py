import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import math
import pandas as pd
from pprint import pprint

style.use('ggplot')

DATA_DIR = "data"
filename = "train.csv"
file = DATA_DIR + "/" + filename

df = pd.read_csv(file)

num_users = max(df.User)
num_tracks = max(df.Track)
print(num_users, num_tracks)

ratings = np.zeros((num_users, num_tracks))

TB = np.matrix(df)

for i in range(len(TB)):
    user = TB[i, 0]
    track = TB[i, 1]
    rating = TB[i, 3]
    ratings[user - 1, track - 1] = rating

pprint(ratings)