import numpy as np
import librosa as lbr
import tensorflow.keras.backend as K
import os

GENRES = [1,2,3,4,5,6,7,8] #list of genres

WINDOW_SIZE = 2048
WINDOW_STRIDE = WINDOW_SIZE // 2
N_MELS = 128
DATASET_PATH = "data/preprocessing"

MEL_KWARGS = {
    'n_fft': WINDOW_SIZE, #window sizes,
    'hop_length': WINDOW_STRIDE,
    'n_mels': N_MELS
}

def load_track(filename, enforce_shape = None):
    '''
        Read song from provided path
        generate melspectrogram
        enforce it to enforce_shape
        none data space will be replaced by 1e-6
    '''

    new_input, sample_rate = lbr.load(filename)
    melspectro = lbr.feature.melspectrogram(new_input, **MEL_KWARGS).T

    if(enforce_shape is not None):
        if(melspectro.shape[0] < enforce_shape[0]):
            delta_shape = (enforce_shape[0] - melspectro.shape[0], 
                           enforce_shape[1])
            melspectro = np.append(melspectro, np.zeros(delta_shape), axis = 0)
        elif melspectro.shape[0] > enforce_shape[0]:
            melspectro = melspectro[: enforce_shape[0], :]
    
    melspectro[melspectro == 0] = 1e-6
    
    # np.save(os.path.join(DATASET_PATH, filename), (np.log(melspectro), float(new_input.shape[0])/sample_rate))
    return (np.log(melspectro), float(new_input.shape[0])/sample_rate)

def get_layer_output_function(model, layer_name):
    # get input of input layer
    input = model.get_layer('input').input

    # get output of the targeted layer
    output = model.get_layer(layer_name).output

    # get output of the targeted layer
    f = K.function([input, K.learning_phase()], [output])

    return lambda x : f([x, 0])[0]