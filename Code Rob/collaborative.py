import pandas as pd
import numpy as np
import json
import _pickle as pickle
from time import sleep
import math

DIR = "data/"
filename = "train.csv"
file = DIR + filename

df = pd.read_csv(file, index_col=2)

ratings = list(df.Rating)
mean_rating = np.mean(ratings)

users = list(df.index.values)
num_users = len(users)

with open('data.p', 'rb') as fp:
    R = pickle.load(fp)

print(R)

def weight(orig, comp, rating):
    if orig < mean_rating:
        if comp > mean_rating:
            pass
        if comp < mean_rating:
            sim = 1 - np.abs(orig - comp) / mean_rating
            print(orig, comp, sim)
            return sim * rating
    if orig > mean_rating:
        if comp > mean_rating:
            sim = 1 - np.abs(orig - comp) / (100 - mean_rating)
            print(orig, comp, sim)
            return sim * rating
        if comp < mean_rating:
            pass

def check_users(rating, t):
    for i in range(len(R)):
        pass

def predict(user, t):
    comp_ratings = []
    if user in users:
        ratings = dict()
        records = df.loc[user]
        for i in range(len(records)):
            track, rating = records.iloc[i].Track, records.iloc[i].Rating

def predict2(user, t):
    comp_ratings = []
    if user in users:
        ratings = df.loc[user]

        ## LOOP OVER RATED SONGS BY USER 1 ##
        for index, track in ratings.iterrows():
            song = track["Track"]
            rating = track["Rating"]

            ## GET ALL USERS THAT HAVE ALSO RATED THIS SONG
            shared = df.loc[df["Track"] == song]
            target = df.loc[shared.index.values]

            ## GET ALL USERS THAT HAVE ALSO RATED THE SONG OF INTEREST ##
            target_ratings = target.loc[target["Track"] == t]
            for user2, track2 in target_ratings.iterrows():
                orig = rating
                ratings2 = df.loc[user2]
                comp = float(ratings2.loc[ratings2["Track"] == song].Rating)
                rating2 = track2["Rating"]
                weight(orig, comp, rating2)

predict2(667, 3)