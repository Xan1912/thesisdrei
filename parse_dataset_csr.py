import pandas as pd

def clean_text(each):
    each = each.replace('[','').replace(']','').replace('\"','').replace('\'','').replace(',','.').lower()+'.'
    return each

# Load mustard
df_mstrd = pd.read_csv("all_mustard.tsv", delimiter='\t', header=0, usecols=["utterance","sarcasm"])
# Load mustard csr tsv
df_csr = pd.read_csv("all_mustard_csr_clean.tsv", delimiter='\t', header=0, usecols=["sentence","utterance"])
df_csr["combined"] = df_csr["sentence"]+df_csr["utterance"].apply(lambda row: clean_text(row))
df = pd.concat([df_csr, df_mstrd], axis=1)

sentences = df.combined.values
labels = df.sarcasm.values
df_csr['utterance'] = df_csr["utterance"].apply(lambda row: clean_text(row))
csr_only = df_csr.utterance.values
