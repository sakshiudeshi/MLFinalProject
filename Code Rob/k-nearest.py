import pandas as pd
import numpy as np
import operator
from pprint import pprint

DATA_DIR = "data"
filename = "users.csv"
file = DATA_DIR + "/" + filename

df = pd.read_csv(file, index_col=0)
df = df.sort_index()

users = list(df.index.values)
num_users = len(users)

genders = df.GENDER
ages = df.AGE
music = df.MUSIC

num_questions = 19

Q = []

for i in range(num_questions):
    Q.append("Q" + str(i + 1))

questions = df[Q].fillna(0)

def k_nearest(user, k):
    X = np.matrix(questions.iloc[user])
    other = np.matrix(questions)
    Y = np.linalg.norm(X - other, axis=1)
    dist = {}
    for i in range(num_users):
        dist[users[i]] = Y[i]
    dist = sorted(dist.items(), key=operator.itemgetter(1))
    k_users = [x[0] for x in dist[1:k + 1]]
    k_distances = [x[1] for x in dist[1:k + 1]]

    return k_users, k_distances

k_users, k_distances = k_nearest(user = 3, k = 10)
print(k_users)
print(k_distances)
