import pandas as pd
import csv
import re

def clean_text(each):
    each = each.replace('[','').replace(']','').replace('\"','').replace('\'','').replace('','').lower()
    return each

def split_string(each):
    each = re.sub(r'(0|([0-9][0-9])|([0-9][0-9])|[1-9])\s{6}', 'a', each)
    return each

all_csr = list()
for i in range(7):
    df_csr = pd.read_csv("all_mustard_csr{}.tsv".format(i+1), delimiter='\t', header=0, usecols=["utterance"])
    df_csr['sentence'] = df_csr['utterance'].apply(lambda x: clean_text(x) if x!='[]' else ' ')
    if i>0:
        all_csr.append(df_csr[1:])
    else:
        all_csr.append(df_csr)

with open('all_mustard.tsv', 'w') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['csr_utterance'])
    for each in all_csr:
        current = each['sentence'].values
        for i in range(len(current)):
            sent = current.tolist()[i]
            tsv_writer.writerow([sent])
            
out_file.close()