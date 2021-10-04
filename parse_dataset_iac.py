from pathlib import Path
import pandas as pd

# main dataset folder ; path to the IAC folder ; define IAC path
data_folder = Path("datasets/") ; path_to_IAC = data_folder / "IAC/sarcasm_v2/" ; IAC_GEN = path_to_IAC / "GEN-sarc-notsarc.csv"
# Load the dataset into a pandas dataframe.
df = pd.read_csv(IAC_GEN, header=0, usecols=["label","text"])
# change data to boolean
df['label'] = df['label'].apply(lambda x: True if x == 'sarc' else False)
# Report the number of sentences.
print('Number of test sentences: {:,}\n'.format(df.shape[0]))
sentences = df.text.values
labels = df.label.values