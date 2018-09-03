import csv
import pickle
import numpy as np
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.optimizers import RMSprop

if __name__ == '__main__':

    #load model
    with open('model/model.yaml', 'r') as f:
        model = model_from_yaml(f.read())
    model.load_weights('model/model_weight.h5')
    
    print('Loaded model.')
    
    opt = RMSprop(lr=0.00001)
    model.compile(
        loss = 'categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )
    
    #get data
    metadata = pickle.load(open('data/test.pickle', 'rb'))
    x_test = metadata['x']


    # model.predict(x_test)
    
    #evaluate
    print('Evaluate model.')
    result = {'y': model.predict(x_test), 'y_name': metadata['filename']}

    with open('data/sample_submission.csv', 'r') as r, open('result.csv', 'w') as w:

        result['y'] = np.argmax(result['y'], axis=1)
        fieldnames = ['Id', ' Genre']
        writer = csv.DictWriter(w, fieldnames = fieldnames)
        writer.writeheader()
        reader = csv.reader(r)
        next(reader, None)
        #get index of file name => get genre
        for row in reader:
            item_index = np.where(result['y_name'] == row[0])
            writer.writerow({'Id': row[0], ' Genre': int(result['y'][item_index]) + 1})