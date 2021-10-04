import numpy as np
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Flatten
from keras.models import Model
from keras.models import Sequential
from keras.layers import TimeDistributed
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from numpy import zeros
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing
from word_to_vec import w2v
from file_handling_BiLSTM import *
from senticnet6 import senticnet
import wandb
import argparse

############################################## Bi LSTM with word embeddings ##########################################################

def get_polarity_scores(word):
    # 'introspection_value', 'temper_value', 'attitude_value', 'sensitivity_value'
    if word in senticnet.keys():
        print('found')
        introspection_value = float(senticnet[word][0])
        temper_value = float(senticnet[word][1])
        attitude_value = float(senticnet[word][2])
        sensitivity_value = float(senticnet[word][3])
        polarity_value = float(senticnet[word][7])
        return np.array([introspection_value,temper_value,attitude_value,sensitivity_value,polarity_value])
    else:
        return np.array([0.01, 0.01, 0.01, 0.01, 0.01])

class BiLSTM:  
    def __init__(self,emotion_ls, token_ls, utterances, test_size=0.3, val_size=0.10, polarity=0, epochs = 10, max_len=40):
        self.emotion_ls = emotion_ls
        self.token_ls = token_ls
        self.utterances = utterances
        self.test_size = test_size
        self.val_size = val_size
        self.max_len = max_len
        self.polarity = polarity
        self.epochs = epochs

    def train(self):
        le = preprocessing.LabelEncoder()
        self.Y_new = self.emotion_ls  
        self.Y_new = le.fit_transform(self.Y_new)
        if self.Y_new.size > 0:
            self.inverted_label = le.inverse_transform(self.Y_new)
    
        # prepare tokenizer
        self.t = Tokenizer()
        self.t.fit_on_texts(self.token_ls)
        self.vocab_size = len(self.t.word_index) + 1
        encoded_docs = self.t.texts_to_sequences(self.utterances) 
        max_length = self.max_len
        self.X = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
        self.y = self.Y_new

        # get the embedding matrix from the embedding layer
        if self.polarity == 1:
            self.embedding_matrix = zeros((self.vocab_size, 305))
        else:
            self.embedding_matrix = zeros((self.vocab_size, 300))

        # for the word representations
        for word, i in self.t.word_index.items():
            embedding_vector = w2v.get(word)
            if self.polarity == 1:
                pol = get_polarity_scores(word)
                if embedding_vector is not None:
                    # for word embeddings and polarity scores 
                    self.embedding_matrix[i] = np.append(embedding_vector,pol)
                    self.size = self.embedding_matrix[i].shape[0]
            else:
                if embedding_vector is not None:
                    # for only word embeddings
                    self.embedding_matrix[i] = embedding_vector
                    # print(self.embedding_matrix[i].shape)
                    self.size = self.embedding_matrix[i].shape[0]

        # Splitting into test and training data
        self.X_train,self.X_test, self.Y_train, self.Y_test =  train_test_split(self.X, self.y, test_size=self.test_size, random_state=4)
        print('Train:', len(self.X_train))
        print('Test:', len(self.X_test))
        # can pass all these variables as parameters to the model function

    # main model
    def rnn_model(self):
        wandb.init()
        self.model = Sequential()
        input_initial = Input(shape=(self.max_len,))
        self.model = Embedding(self.vocab_size,self.size,weights=[self.embedding_matrix],input_length=self.max_len)(input_initial)
        self.model =  Bidirectional (LSTM (self.size,return_sequences=True,dropout=0.20),merge_mode='concat')(self.model)
        self.model = TimeDistributed(Dense(self.size,activation='relu'))(self.model)
        self.model = Flatten()(self.model)
        self.model = Dense(self.size,activation='relu')(self.model)
        output = Dense(2,activation='softmax')(self.model)
        self.model = Model(input_initial,output)
        self.model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
        train_history = self.model.fit(self.X_train,self.Y_train,validation_split=self.val_size, epochs = self.epochs, verbose = 1)
        loss = train_history.history['loss']
        # val_loss = train_history.history['val_loss']
        train_metrics = {"training_loss": loss}
        wandb.log(train_metrics)
        return self.evaluate()

    def evaluate(self):
        # evaluate the model
        self.Y_pred = self.model.predict(self.X_test)
        self.y_pred = np.array([np.argmax(pred) for pred in self.Y_pred])
        print('Result:\n',classification_report(self.Y_test,self.y_pred),'\n')

preprocess = Preprocess()
labels = preprocess.process_emotions()
utterances = preprocess.process_utterances()
tokens = preprocess.process_token_ls()

# ------------ Arguement Parsers ------------ #
parser = argparse.ArgumentParser(description="Training parameters")
parser.add_argument("test_size", type=float)
parser.add_argument("val_size", type=float)
parser.add_argument("polarity", type=int)
parser.add_argument("epochs", type=int)
args = parser.parse_args()

# ------------ Instanstiate model ------------ #
'''
Sample command to run the script
python3 Desktop/Thesis/bi_LSTM.py 0.3 0.1 0 10
'''
bilstm = BiLSTM(labels, tokens, utterances, args.test_size, args.val_size, args.polarity, args.epochs)
bilstm.train()                                              
bilstm.rnn_model()















































































































































































































































































































