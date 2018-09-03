import csv
import multiprocessing as mp
import os
from optparse import OptionParser
from pickle import dump

import numpy as np

from common import load_track

# default file path
DEFAULT_FILE = "3672629935623490 (copy).mp3"


def get_default_shape():
    tmp_features, _ = load_track(DEFAULT_FILE)
    return tmp_features.shape


def collect_data(metadata, dataset_path):
    '''
        Read song from dataset_path
        Convert song to melspectrogram
        :param dataset_path: path to the dataset
        :return x, y:
    '''

    default_shape = get_default_shape()

    track_count = len(metadata.keys())
    # track_count = 3
    # TODO why + 2 tuples
    x = np.zeros((track_count,) + default_shape, dtype=np.float32)
    # TODO troi tru dat diet
    y = np.zeros((track_count, 10), dtype=np.float32)

    pool = mp.Pool(processes=os.cpu_count())
    results = []
    for index, file_name in enumerate([*metadata]):
        path = os.path.join(dataset_path, file_name)
        results.append(pool.apply_async(load_track, args=(path, default_shape)))

    for index, file_name in enumerate([*metadata][:track_count]):
        print('processing {}/{}'.format(str(index + 1), str(track_count)))
        data = results[index].get()
        x[index] = data[0]
        # x[index][1] = data[1]
        y[index][int(metadata[file_name]) - 1] = 1

    return {'x': x, 'y': y}


parser = OptionParser()
parser.add_option('-t', '--trainmetadata', dest='metadata', default='data/train.csv')
parser.add_option('-d', '--dataset', dest='dataset_path',
                  default="/home/dualeoo/train")
options, args = parser.parse_args()

labelDic = {}

with open(options.metadata, 'r') as f:
    data = csv.reader(f)
    for row in data:
        labelDic[row[0]] = row[1]

data = collect_data(labelDic, options.dataset_path)
print(data)

with open('data/train.pickle', 'wb') as f:
    dump(data, f, protocol=4)

