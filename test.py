from optparse import OptionParser
from common import load_track, get_layer_output_function
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import model_from_yaml, Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import RMSprop

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-m', '--model', dest='model', default = 'model/model.yaml')
    parser.add_option('-w', '--weight', dest='weight', default = 'model/model_weight.h5')
    options, args = parser.parse_args()

    with open(options.model, 'r') as f:
        model = model_from_yaml(f.read())
    model.load_weights(options.weight)
    (features, duration) = load_track('3672629935623490 (copy).mp3')
    features = np.reshape(features, (1,) + features.shape)
    opt = RMSprop(lr=0.00001)
    model.compile(
        loss = 'categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )
    model.evaluate(features, np.array([[0, 0, 0, 0, 0, 0, 1, 0]]))
    print('Loaded model.')