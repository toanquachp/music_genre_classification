from common import load_track, get_layer_output_function

import pickle
from optparse import OptionParser
import numpy as np

from tensorflow.keras.layers import Input
from tensorflow.keras.models import model_from_yaml, Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import RMSprop

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-t', '--testdata', dest='testdata', default='data/test.pickle')
    parser.add_option('-m', '--model', dest='model', default = 'model/model.yaml')
    parser.add_option('-w', '--weight', dest='weight', default = 'model/model_weight.h5')
    options, args = parser.parse_args()

    #load model
    with open(options.model, 'r') as f:
        model = model_from_yaml(f.read())
    model.load_weights(options.weight)
    
    print('Loaded model.')
    
    opt = RMSprop(lr=0.00001)
    model.compile(
        loss = 'categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )
    
    #get data
    metadata = pickle.load(open(options.testdata, 'rb'))
    x_test = metadata['x']
    y_test = metadata['y']
    
    #evaluate
    print('Evaluate model.')
    print(model.evaluate(x_test, y_test))
    