import csv
import cPickle as pickle

word_dict = {}


def get_dict(path):
    d = {}
    with open(path, "rb") as f:
        for line in f:
            (val, key) = line.split(None, 1)
            d[str(key.rstrip())] = int(val)
    return d


heard_of_dict = get_dict("data/words_heard-of.txt")
own_artist_music_dict = get_dict("data/words_own-artist-music.txt")


def get_int_attr(attr):
    if attr:
        return int(attr)
    else:
        return -1


def get_int_attr_dict(attr, dict):
    if attr in dict:
        return int(dict[attr])
    else:
        return -1


def get_flt_attr(attr):
    if attr:
        return float(attr)
    else:
        return -1.0


def read_words(csv_filename):
    csv_file = open(csv_filename, 'rb')
    reader = csv.reader(csv_file)
    reader.next()
    for row in reader:
        artist_id = int(row[0])
        user_id = int(row[1])
        word = {}
        word['heard-of'] = get_int_attr_dict(row[2], heard_of_dict)
        word['own-artist-music'] = get_int_attr_dict(row[3], own_artist_music_dict)
        word['like-artist'] = get_flt_attr(row[4])
        for j in range(81):
            word['w%d' % (j + 1)] = get_int_attr(row[5 + j])
        word_dict[(artist_id, user_id)] = word
    csv_file.close()


def load_words(pkl_filename):
    global word_dict
    pkl_file = open(pkl_filename, 'rb')
    word_dict = pickle.load(pkl_file)
    pkl_file.close()


def save_words(pkl_filename):
    pkl_file = open(pkl_filename, 'wb')
    pickle.dump(word_dict, pkl_file, -1)
    pkl_file.close()


if __name__ == "__main__":
    read_words("data/words.csv")
    save_words('data/words.pkl')
