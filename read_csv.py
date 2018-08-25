import csv
import pickle
# import cPickle

labelDic = {}

with open('train.csv', 'r') as f:
    data = csv.reader(f)
    for row in data:
        labelDic[row[0]] = row[1]

with open('data.pickle', 'wb') as f:
    pickle.dump(labelDic, f, protocol=pickle.HIGHEST_PROTOCOL)
