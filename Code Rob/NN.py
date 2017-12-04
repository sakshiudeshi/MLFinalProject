import numpy as np
import pandas as pd
import math
from random import shuffle

DIR = "data/"
filename = "train.csv"
file = DIR + filename

df = pd.read_csv(file, index_col=2)

users = list(set(df.index.values))
num_users = len(users)

tracks = list(set(df["Track"]))
num_tracks = len(tracks)
n = num_tracks

## RANDOMLY RESHUFFLE INTO TRAIN AND TEST ##
perc = 0.007

m_train = math.ceil(perc * num_users)
m_test = num_users - m_train

X_train = np.zeros((n, m_train))
X_test = np.zeros((n, m_test))

Y_train = np.zeros((n, m_train))
Y_test = np.zeros((n, m_test))

train_users = users[:m_train]
test_users = users[m_train:]

C_train = np.zeros((m_train, n))
C_test = np.zeros((m_test, n))

c = 0
for user in train_users:
    print(str(c) + " out of " + str(m_train))
    records = df.loc[[user]]
    for i in range(len(records)):
        track = records.iloc[i].Track
        rating = records.iloc[i].Rating
        X_train[track - 1][c] = rating
        Y_train[track - 1][c] = rating
    c += 1

print(X_train.shape, Y_train.shape)

## DEFINE NUMBER OF HIDDEN LAYERS AND HIDDEN UNITS ##
weights = [["relu", 20], ["relu", 30], ["relu", n]]
L = len(weights)

## INITIALIZE WEIGHTS ##
W = list()
b = list()

weight = np.random.randn(weights[0][1], n) * 0.01
bias = np.zeros((weights[0][1], 1))

W.append(weight)
b.append(bias)

for i in range(1, len(weights)):
    weight = np.random.randn(weights[i][1], W[i - 1].shape[0]) * 0.01
    bias = np.zeros((weights[i][1], 1))
    W.append(weight)
    b.append(bias)

def cost(A, Y, C):
    m =  sum(sum(C))
    cost = -1 / m * (C*np.dot(Y, np.log(A).T) + C*np.dot(1 - Y, np.log(1 - A).T))
    return cost

def forward(X, W, b, learning_rate = 0.01):
    A = X
    for l in range(L):
        Z = np.dot(W[l], A) + b[l]
        if weights[l][0] == "relu":
            A = np.maximum(Z, 0)
        if weights[l][0] == "softmax":
            A = np.exp(Z) / sum(np.exp(Z))

    return A








