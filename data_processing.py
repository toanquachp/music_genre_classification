import csv
from sklearn.model_selection import train_test_split
from common import GENRES, load_track
import sys
import numpy as np
from math import pi
import pickle
from pickle import dump
import os
from optparse import OptionParser

#default file path
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
    
    x = np.zeros((track_count, ) +  default_shape, dtype = np.float32)
    y = np.zeros((track_count, len(GENRES)), dtype=np.float32)

    for index, file_name in enumerate([*metadata][:track_count]):
        print('processing {}/{}'.format(str(index + 1), str(track_count)))
        
        path = os.path.join(dataset_path, file_name)
     
        data = load_track(path, default_shape)
        
        x[index] = data[0]
        # x[index][1] = data[1]
        y[index][int(metadata[file_name])-1] = 1
        
    (x_train, x_val, y_train, y_val) = train_test_split(x, y, test_size=0.3)
    
    return {'x_train': x_train, 'x_val': x_val, 
            'y_train': y_train, 'y_val': y_val}


parser = OptionParser()
parser.add_option('-t', '--trainmetadata', dest='metadata', default = 'data/train.csv')
parser.add_option('-d', '--dataset', dest='datasetpath', default = "/home/shiro/Projects/MusicGeneration/data/before/train")
parser.add_option('-o', '--output', dest='output', default='data/')

options, args = parser.parse_args()

labelDic = {}

with open(options.metadata, 'r') as f:
    data = csv.reader(f)
    for row in data:
        labelDic[row[0]] = row[1]

data = collect_data(labelDic, options.dataset_path)

# data = collect_data('/home/shiro/Projects/MusicGeneration/CRNN - Live Music Genre Recognition/data.pickle')

with open(options.output + 'data_processing.pickle', 'wb') as f:
    dump(data, f)