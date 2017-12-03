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

## DEFINE THE AMOUNT OF HIDDEN CLUSTERS FOR USERS AND SONG ##
c = 5   # number of user clusters
d = 10  # number of song clusters

## INITIALIZE POSTERIORS TO BE EQUAL ##
PZy = np.matrix(c * [1/c])
PZx = np.matrix(d * [1/d])

PyZy = np.matrix(L * [c * [1/c]])
PxZx = np.matrix(L * [d * [1/d]])
PrZxZy = np.zeros((c, d, num_ratings))
PrZxZy.fill(1 / num_ratings)

denom = 0
for x in range(d):
    for y in range(c):
        denom += PZx[x] * PZy[y] * PxZx[x] * PyZy[y] * PrZxZy

for x in range(d):
    new_posteriors = c * []
    for y in range(c):
        num =  PZx[x] * PZy[y] * PxZx[x] * PyZy[y] * PrZxZy





