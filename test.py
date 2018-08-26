from common import load_track, get_layer_output_function
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import model_from_yaml, Model
from tensorflow.keras import backend as K

with open('model.yaml', 'r') as f:
    model = model_from_yaml(f.read())
model.load_weights('model_weight.h5')
pred_fun = get_layer_output_function(model, 'output_merged')

(features, duration) = load_track('3672629935623490 (copy).mp3')
features = np.reshape(features, (1,) + features.shape)

print('Loaded model.')
print(pred_fun(features))