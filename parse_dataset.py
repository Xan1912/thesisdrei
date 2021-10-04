import pandas as pd

# Load the dataset into a pandas dataframe.
df = pd.read_csv("all_mustard.tsv", delimiter='\t', header=0, usecols=["utterance","sarcasm"])
# Report the number of sentences.
print('Number of training sentences: {:,}\n'.format(df.shape[0]))
# Display 10 random rows from the data.
# print(df.sample(10))
# Get the lists of sentences and their labels.
sentences = df.utterance.values
labels = df.sarcasm.values



