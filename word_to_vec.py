import numpy as np

embedding_path = "/Users/user/Documents/glove.6B/glove.6B.300d.txt" 

def get_word2vec(embedding_path):
    __file = open(embedding_path, "r")
    if (__file):
        word2vec = dict()
        split = __file.read().splitlines()
        for line in split:
            key = line.split(' ',1)[0] # the first word is the key
            value = np.array([float(val) for val in line.split(' ')[1:]])
            word2vec[key] = value
        return (word2vec)
    else:
        print("Invalid file path")

w2v = get_word2vec(embedding_path)