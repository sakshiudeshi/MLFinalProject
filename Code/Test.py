import csv
import os


def csv_reader(file_obj):
    reader = csv.reader(file_obj)
    data = []
    for row in reader:
        data.append(" ".join(row))
    return data

def sanitize(path):
    data_set = []
    with open(path, "rb") as f_obj:
        data = csv_reader(f_obj)

    for item in data:
        data_set.append(item)
    return data_set

if __name__ == '__main__':
    train_path = os.getcwd() + "/data/train.csv"
    print sanitize(train_path)[0]