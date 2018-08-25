from tensorflow.keras.layers import Input, Dense, \
    Lambda, Dropout, Activation, \
    LSTM, TimeDistributed, Convolution1D, \
    MaxPooling1D
from tensorflow.keras.optimizers import RMSprop
from common import GENRES, load_track
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import pickle

N_LAYERS_CONV = 3
FILTER_LENGTH = 5
CONV_FILTER_COUNT = 256
LSTM_COUNT = 256
BATCH_SIZE = 32
EPOCH_COUNT = 10

def train_model(data):
    x_train = data['x']
    y_train = data['y']

    n_features = x_train.shape[2]
    input_shape = (None, n_features)

    model_input = Input(input_shape, name="input")
    layer = model_input

    for i in range(N_LAYERS_CONV):
        layer = Convolution1D(
                filters = CONV_FILTER_COUNT,
                kernel_size = FILTER_LENGTH,
                name = 'convolution_' + str(i + 1)
                )(layer)
        layer = Activation('relu')(layer)
        layer = MaxPooling1D(2)(layer)

    layer = Dropout(0.5)(layer)
    layer = LSTM(LSTM_COUNT, return_sequences=True)(layer)
    layer = Dropout(0.5)(layer)
    layer = TimeDistributed(Dense(len(GENRES)))(layer)
    layer = Activation('softmax', name='output_realtime')(layer)

    time_distributed_merge_layer = Lambda(
            function=lambda x: K.mean(x, axis=1), 
            output_shape=lambda shape: (shape[0],) + shape[2:],
            name='output_merged'
        )

    model_output = time_distributed_merge_layer(layer)

    model = Model(model_input, model_output)
    opt = RMSprop(lr=0.00001)

    model.compile(
        loss = 'categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )
    print('Training...')
    
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH_COUNT,
              validation_data=(x_train, y_train), verbose=1)

    return model

metadata = pickle.load(open('data_processing.pickle', 'rb'))
model = train_model(metadata)

to_predict = load_track("/home/shiro/Projects/MusicGeneration/CRNN - Live Music Genre Recognition/data/before/train/3626043815719777.mp3")
model.predict(to_predict)
