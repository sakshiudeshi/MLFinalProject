import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

DIR = "data/"
filename = "train.csv"
file = DIR + filename

df = pd.read_csv(file)
#df = df[:100]

## PREPARE ARRAYS OF SONGS, USERS and RATINGS ##
X = np.array(df.User)   #SONGS
Y = np.array(df.Track)  #USERS
R = np.array(df.Rating) #RATINGS

assert len(X) == len(Y) == len(R)
L = len(X)

users = set(X)
tracks = set(Y)
ratings = set(R)

num_users = max(users)
num_tracks = max(tracks)
num_ratings = max(ratings)

## DEFINE THE AMOUNT OF HIDDEN CLUSTERS FOR USERS AND SONG ##
c = 5   # number of user clusters
d = 10  # number of song clusters

## INITIALIZE POSTERIORS TO BE EQUAL ##
PZy = np.random.rand(c, 1)
PZy = np.matrix(PZy / np.sum(PZy))

PZx = np.random.rand(d, 1)
PZx = np.matrix(PZx / np.sum(PZx))

PyZy = np.random.rand(num_users, c)
PyZy = np.matrix(PyZy / np.sum(PyZy))

PxZx = np.random.rand(num_tracks, d)
PxZx = np.matrix(PxZx / sum(PxZx))

PrZxZy = np.random.rand(num_ratings, d, c)
PrZxZy = PrZxZy / sum(PrZxZy)

P = np.zeros((L, d, c))

def get_prob(user, track, rating):
    p = 0
    p = np.sum(np.multiply(np.dot(PZx, PZy.T), np.dot(PxZx[track - 1].T, PyZy[user - 1])))
    return p

def log_prob():
    p = 0
    for l, row in df.iterrows():
        user, track, rating = row.User, row.Track, row.Rating
        p += get_prob(user, track, rating)

    return p


def expectation(P, b):
    for l, row in df.iterrows():
        user, track, rating = row.User, row.Track, row.Rating
        P[l] = np.multiply(np.dot(PZx, PZy.T), np.dot(PxZx[track - 1].T, PyZy[user - 1]))
        P[l] /= np.sum(P[l])

    return P

def maximization(PZx, PZy, PxZx, PyZy, PrZxZy):
    PZx = np.matrix(np.sum(np.sum(P, axis=2), axis=0) / L).T
    PZy = np.matrix(np.sum(np.sum(P, axis=1), axis=0) / L).T
    for track in range(num_tracks):
        if track in tracks:
            Ls = df.loc[df.Track == track]
            Ls = Ls.index.values
            PxZx[track - 1] = np.sum(np.sum(P[Ls], axis=2), axis=0)
    PxZx = PxZx / np.sum(PxZx, axis=0)

    for user in range(num_users):
        if user in users:
            Ls = df.loc[df.User == user]
            Ls = Ls.index.values
            PyZy[user - 1] = np.sum(np.sum(P[Ls], axis=1), axis=0)
    PyZy = PyZy / np.sum(PyZy, axis=0)

    for rating in range(num_ratings):
        if rating in ratings:
            Ls = df.loc[df.Rating == rating]
            Ls = Ls.index.values
            PrZxZy[rating - 1] = np.sum(P[Ls], axis=0)
    PrZxZy = PrZxZy / np.sum(PrZxZy, axis=0)

    return PZx, PZy, PxZx, PyZy, PrZxZy

iterations = 100

LP = []
for iteration in range(iterations):
    print("Iteration ", iteration, " out of ", iterations)
    P = expectation(P, 1)
    PZx, PZy, PxZx, PyZy, PrZxZy = maximization(PZx, PZy, PxZx, PyZy, PrZxZy)
    LP.append(log_prob())
    print(PZx.T)
    if len(LP) > 1:
        if (LP[iteration] - LP[iteration - 1]) <= 0:
            break

plt.plot(LP)
plt.show()







