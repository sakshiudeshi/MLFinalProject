import pandas as pd
import numpy as np
from pprint import pprint
from datetime import datetime, date

DIR = "data/"
filename = "train.csv"
file = DIR + filename

df = pd.read_csv(file)

## PREPARE ARRAYS OF SONGS, USERS and RATINGS ##
X = np.array(df.User)   #SONGS
Y = np.array(df.Track)  #USERS
R = np.array(df.Rating) #RATINGS

assert len(X) == len(Y) == len(R)
L = len(X)

users = set(X)
tracks = set(Y)
ratings = set(R)

num_users = len(users)
num_tracks = len(tracks)
num_ratings = len(ratings)

print("num_users: " + str(num_users))
print("num_track: " + str(num_tracks))

## DEFINE THE AMOUNT OF HIDDEN CLUSTERS FOR USERS AND SONG ##
c = 5   # number of user clusters
d = 10  # number of song clusters

## INITIALIZE POSTERIORS TO BE EQUAL ##
PZy = c * [1/c]
PZx = d * [1/d]

PyZy = num_users * [c * [1/c]]
PxZx = num_tracks * [d * [1/d]]
PrZxZy = d * [c * [num_ratings * [1 / num_ratings]]]

P = []

def pr(zx, zy, x, y, r):
    p = []
    for i in range(d):
        for j in range(c):
            p.append(PZx[i] * PZy[j] * PxZx[x][i] * PyZy[y][j] * PrZxZy[i][j][r])

    print(p)
    print(sum(p))








