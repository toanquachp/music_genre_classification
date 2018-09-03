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

def collect_data(data, dataset_path):
    default_shape = get_default_shape()
    track_count = len(data)

    x = np.zeros((track_count,) + default_shape, dtype=np.float32)

    pool = mp.Pool(processes=os.cpu_count())

    results = []
    for index, filename in enumerate([*data]):
        path = os.path.join(dataset_path, filename)
        results.append(pool.apply_async(load_track, args = (path, default_shape)))

    for index, filename in enumerate([*data]):
        print('Processing {}/{}'.format(str(index + 1), str(track_count)))

        x[index] = results[index].get()[0]

    return {'x': x, 'filename': np.array(data)}
    

parser = OptionParser()
parser.add_option('-t', '--testmetadata', dest='metadata', default='data/test.csv')
parser.add_option('-d', '--dataset', dest='dataset_path', default='/home/dualeoo/test')

options, args = parser.parse_args()

name_list = []

data = None

with open(options.metadata, 'r') as f:
    for row in csv.reader(f):
        name_list.append(row[0])


data = collect_data(name_list, options.dataset_path)

with open('data/test.pickle', 'wb') as f:
    dump(data, f, protocol=4)

