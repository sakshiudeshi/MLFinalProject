import pandas as pd
import numpy as np
import json
import _pickle as pickle
from time import sleep
import math
import operator
from pprint import pprint
import csv
from datetime import datetime, date

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
cache = {}

def k_nearest(user, k, target_users):
    if user not in users:
        return None

    if user not in cache:
        X = np.matrix(questions.loc[user])
        other = np.matrix(questions)
        Y = np.linalg.norm(X - other, axis=1)
        dist = []
        for i in range(num_users):
            dist.append([users[i], Y[i]])

        cache[user] = dist
    else:
        print("Using cached distances!")

    dist = cache[user]

    dist = [x for x in dist if x[0] in target_users]
    dist = sorted(dist, key=lambda x: x[1])

    k_users = [x[0] for x in dist[1:k + 1]]
    k_distances = [x[1] for x in dist[1:k + 1]]

    return k_users, k_distances

## PREDICT RATING FROM TRAIN DATA ##

DIR = "data/"
filename = "train.csv"
file = DIR + filename

df = pd.read_csv(file, index_col=2)

with open('data.p', 'rb') as fp:
    R = pickle.load(fp)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def predict(user, t):
    k = 500

    target = df.loc[df["Track"] == t]
    target_users = sorted(target.index.values)
    comp_users, distances = k_nearest(user, k, target_users)

    weights = softmax(distances)

    ratings = target.loc[comp_users].Rating
    prediction = sum(weights[::-1] * ratings)

    return prediction

def MRSE():
    file = "predictions.csv"
    with open(file, 'w', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["User", "Rating", "Prediction", "MSE"])

    predictions = list()
    MSE_lst = []
    i = 0
    I = len(df)
    avg = 0

    print_int = 10
    save_int = 1000
    for user, data in df.iterrows():
        start = datetime.time(datetime.now())
        i += 1

        if user in users:
            track = data["Track"]
            rating = data["Rating"]
            prediction = predict(user, track)
            MSE = np.abs(rating - prediction)**2
            predictions.append([user, rating, prediction, MSE])

            MSE_lst.append(MSE)

            end = datetime.time(datetime.now())
            duration = (datetime.combine(date.min, end) - datetime.combine(date.min, start)).total_seconds()
            avg = 0.9 * avg + 0.1 * duration

        if i % print_int == 0:
            print("Checked " + str(i) + " out of " + str(I) + " rows")
            RMSE = np.sqrt(np.mean(MSE_lst))
            print("RMSE so far: " + str(RMSE))
            left = round((I - i) * avg / 3600, 2)
            print("Estimated amount of hours left: " + str(left) + "\n")
        if i % save_int == 0:
            file = "predictions.csv"
            with open(file, 'a', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                for line in predictions:
                    writer.writerow(line)

            predictions = list()
            print("Saved! \n")

    RMSE = np.sqrt(np.mean(MSE_lst))
    return RMSE, predictions

RMSE, predictions = MRSE()

print("RMSE: " + str(RMSE))

file = "predictions.csv"

with open(file, 'a', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    for line in predictions:
        writer.writerow(line)