import pandas as pd

def clean_text(each):
    each = each.replace('..','.').replace('[','').replace(']','').replace('\"','').replace('\'','').replace(',','.').lower()+'.'
    return each

all_csr = list()
# Load the reasoning dataset 
for i in range(7):
    df_csr = pd.read_csv("all_mustard_csr{}.tsv".format(i+1), delimiter='\t', header=0, usecols=["sentence","utterance"])
    df_csr["utterance"] = df_csr["utterance"].apply(lambda row: clean_text(row))
    all_csr.append(df_csr)

df_csr = pd.concat(all_csr, ignore_index=True)
print('Obtaining data from reasoning ...')
# Get the lists of sentences and their labels.
sentences = df_csr.utterance.values

