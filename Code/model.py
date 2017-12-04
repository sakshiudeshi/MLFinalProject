from math import sqrt
from collections import defaultdict
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

import users
import words


users.load_users('data/users.pkl')
words.load_words('data/words.pkl')


def indicator(cat_id, dim):
    v = [0] * dim
    if cat_id >= 0:
        v[cat_id] = 1
    return v


def binary(w):
    if w == -1:
        return 0
    if w == 0:
        return -1
    if w == 1:
        return 1


def represent(example):
    vector = []
    artist_id = example['artist']
    track_id = example['track']
    user_id = example['user']
    time_id = example['time']
    vector.extend(indicator(artist_id, 50))
    vector.extend(indicator(track_id, 184))
    vector.extend(indicator(time_id, 24))
    user = users.user_dict.get(user_id, defaultdict(lambda: -1))
    vector.append(binary(user['gender']))
    vector.append(user['age'] / 100.0)
    vector.extend(indicator(user['working'], 13))
    vector.extend(indicator(user['region'], 4))
    vector.extend(indicator(user['music'], 5))
    vector.extend([user['q%d' % (j + 1)] / 100.0 for j in range(19)])
    word = words.word_dict.get((artist_id, user_id), defaultdict(lambda: -1))
    vector.extend(indicator(word['heard-of'], 4))
    vector.extend(indicator(word['own-artist-music'], 5))
    vector.append(word['like-artist'] / 100.0)
    vector.extend([binary(word['w%d' % (j + 1)]) for j in range(81)])
    vector.append(user_id / 50928.0)

    return vector


def label(example):
    return example['rating']

clf = RandomForestRegressor(n_estimators=20, max_features='sqrt')


def validate(examples):
    X = [represent(example) for example in examples]
    y = [label(example) for example in examples]
    scores = cross_val_score(clf, X, y, cv=10, scoring='neg_mean_squared_error')
    return scores


if __name__ == "__main__":
    import music
    train_examples = music.load_examples('data/train.pkl')

    print train_examples[0]

    scores = validate(train_examples)
    print "RMSE: %0.6f (+/- %0.6f)" % (sqrt(-1*scores.mean()), scores.std()/2)
