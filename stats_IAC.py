from pathlib import Path
import pandas as pd
from nltk import word_tokenize

# main dataset folder 
data_folder = Path("datasets/")

# path to the IAC folder
path_to_IAC = data_folder / "IAC/sarcasm_v2/"

# define IAC paths
IAC_GEN = path_to_IAC / "GEN-sarc-notsarc.csv" ; IAC_HYP = path_to_IAC / "HYP-sarc-notsarc.csv" ; IAC_RQ = path_to_IAC / "RQ-sarc-notsarc.csv"
# all the IAC files in a list
files = [IAC_GEN,IAC_HYP,IAC_RQ]
count = 0 ; count_label_ls = list()
for __file in files: 
    # find count of data points
    df = pd.read_csv(__file)
    for col_text in df['class']:
        if col_text == 'sarc':
            count+=1
    count_label_ls.append(count)
__sum = sum(count_label_ls)


## TOTAL DATA POINTS IN IAC is 23590 and sarcasm and not sarcasm share equal value ## 
## find all vocabs in sarcasm

vocab = set() ; count = 0
for __file in files:
    df = pd.read_csv(__file, usecols=['class', 'text'])
    for label, sent in zip(df['class'], df['text']):
        sent = word_tokenize(sent)
        for each in sent: 
            if len(each) < 10 and each not in vocab:
                vocab.add(each)        

print(vocab)




### I have a lot of text data to train 
### For transfer learning I can use Humour datasets, train on positive/negative datasets in news 
### News headlines will have a lot of common sense knowledge / topical awareness 
### Generating common sense graphs from news headlines can be a nice experiment 









