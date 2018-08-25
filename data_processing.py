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
DATASET_PATH = "data/before/train"

def get_default_shape():
    tmp_features, _ = load_track(DEFAULT_FILE)
    return tmp_features.shape

def collect_data(metadata_path):
    '''
        Read song from dataset_path
        Convert song to melspectrogram
        :param dataset_path: path to the dataset
        :return x, y:
    '''
    
    default_shape = get_default_shape()
    print(default_shape)
    metadata = pickle.load(open(metadata_path, 'rb'))
    
    track_count = 10
    
    x = np.zeros((track_count, ) +  default_shape, dtype = np.float32)
    y = np.zeros((track_count, len(GENRES)), dtype=np.float32)

    for index, file_name in enumerate([*metadata][:5]):
        print('processing {}/{}'.format(str(index + 1), str(track_count)))
        
        path = os.path.join(DATASET_PATH, file_name)
     
        data = load_track(path, default_shape)
        
        x[index] = data[0]
        # x[index][1] = data[1]
        y[index][int(metadata[file_name])-1] = 1
        print(y)
        
    return {'x': x, 'y': y}


#TODO: split data to train and validation 8-2
#TODO: save data 
data = collect_data('/home/shiro/Projects/MusicGeneration/CRNN - Live Music Genre Recognition/data.pickle')

with open('data_processing.pickle', 'wb') as f:
    dump(data, f)