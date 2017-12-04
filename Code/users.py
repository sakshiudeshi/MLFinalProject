import csv
import cPickle as pickle

user_dict = {}


def get_dict(path):
    d = {}
    with open(path, "rb") as f:
        for line in f:
            (val, key) = line.split(None, 1)
            d[str(key.rstrip())] = int(val)
    return d


gender_dict = get_dict("data/users_gender.txt")
music_dict = get_dict("data/users_music.txt")
region_dict = get_dict("data/users_region.txt")
working_dict = get_dict("data/users_working.txt")


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


def read_users(csv_path):
    file = open(csv_path, 'rb')
    reader = csv.reader(file)
    reader.next()

    for row in reader:
        id = int(row[0])
        user = {}
        user['gender'] = get_int_attr_dict(row[1], gender_dict)
        user['age'] = get_int_attr(str(row[2]))
        user['working'] = get_int_attr_dict(row[3], working_dict)
        user['region'] = get_int_attr_dict(row[4], region_dict)
        user['music'] = get_int_attr_dict(row[5], music_dict)
        for j in range(19):
            user['q%d' % (j + 1)] = get_flt_attr(row[8 + j])
        user_dict[id] = user

    file.close()


def load_users(pkl_filename):
    global user_dict
    pkl_file = open(pkl_filename, 'rb')
    user_dict = pickle.load(pkl_file)
    pkl_file.close()


def save_users(pkl_filename):
    pkl_file = open(pkl_filename, 'wb')
    pickle.dump(user_dict, pkl_file, -1)
    pkl_file.close()


if __name__ == "__main__":
    read_users('data/users.csv')
    save_users('data/users.pkl')
