import pandas as pd
from stop_words import get_stop_words
from nltk.tokenize import TweetTokenizer
sw = get_stop_words('english')

class Preprocess:
        def __init__(self,file_to_process = "Desktop/Thesis/all_mustard.tsv"):
                self.file_to_process = file_to_process

        def get_tokens(self,sentence):
                self.sentence = sentence
                if type(self.sentence) != float:
                        tknzr = TweetTokenizer()
                        tokens = tknzr.tokenize(self.sentence)
                        # tokens = [token for token in tokens if (token not in sw and len(token) > 1)]
                        # tokens = [token for token in tokens]
                        if type(tokens) != None:
                                return (tokens)

        def process_emotions(self):
                self.df = pd.read_csv(self.file_to_process, delimiter='\t', header=0, usecols=["utterance","sarcasm"])
                labels = self.df.sarcasm.values
                self.sentences = self.df.utterance.values 
                self.token_list = self.df.utterance.apply(lambda row: self.get_tokens(row))
                return labels

        def process_utterances(self):
                return self.sentences
        
        def process_token_ls(self):
                return self.token_list


# preprocess = Preprocess()
# labels = preprocess.process_emotions()
# sentences = preprocess.process_utterances()
# tokens = preprocess.process_token_ls()