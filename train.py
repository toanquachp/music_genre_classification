import pickle

file = open('data.pickle', 'rb')
p = pickle.load(file)
print(p)