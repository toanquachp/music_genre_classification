import pickle


from sklearn.model_selection import train_test_split

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, \
    Lambda, Dropout, Activation, \
    LSTM, TimeDistributed, Convolution1D, \
    MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop


from common import GENRES

N_LAYERS_CONV = 7
FILTER_LENGTH = 5
CONV_FILTER_COUNT = 512
LSTM_COUNT = 256
BATCH_SIZE = 128
EPOCH_COUNT = 96

def train_model(data):
    x = data['x']
    y = data['y']

    (x_train, x_val, y_train, y_val) = train_test_split(x, y, test_size=0.1)

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

    layer = Dropout(0.6)(layer)
    layer = LSTM(LSTM_COUNT, return_sequences=True)(layer)
    layer = Dropout(0.6)(layer)
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
              validation_data=(x_val, y_val), verbose=1)

    return model

if __name__ == "__main__":

    metadata = pickle.load(open('data/train.pickle', 'rb'))

    model = train_model(metadata)

    with open('model/model.yaml', 'w') as f:
        f.write(model.to_yaml())

    model.save_weights('model/model_weight.h5')
