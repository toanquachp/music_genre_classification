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

    x = np.zeroes((track_count, ) + default_shape, dtype = np.float32)

    pool = mp.Pool(processes=os.cpu_count())

    for index, filename in enumerate(data):
        path = os.path.join(dataset_path, filename)
        x[index] = pool.apply_async(load_track, args = (path, default_shape))

    return {'x': x, 'filename': filename}
    

parser = OptionParser()
parser.add_option('-t', '--testmetadata', dest='metadata', default='data/test.csv')
parser.add_option('-d', '--dataset', dest='dataset_path', default='home/dualeoo/test')
parser.add_option('o', '--output', dest='output', default='data/test.pickle')

options, args = parser.parse_args()

name_list = []

data = None

with open(options.metadata, 'rb') as f:
    data = np.array(csv.reader(f))
    print(data)

data = collect_data(data, options.dataset_path)

print(data)

with open(options.output, 'wb'):
    dump(data, protocol=4)
    
