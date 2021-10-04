'''
Data preprocessing file for commonsense 
'''

import csv
import pandas as pd

def check_duplicates(ls):
    result = list()
    for i in range(len(ls)):
        current = ls[i]
        current = current.split(' ')
        first = current[0]
        last = current[-1]
        
        duplicate = last+' is related to '+first
        if duplicate not in result:
            result.append(' '.join(current))

    return result

def check_same_syntax(ls):
    result = list()
    J=list()
    for i in range(len(ls)):
        current = ls[i]
        current = current.split(' ')
        first = current[0]
        last = current[-1]
        
        duplicate = last+' is related to '+first
        if duplicate not in result:
            result.append(' '.join(current))

    return result

with open('all_mustard_csr_clean.tsv', 'w') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['sentence', 'utterance'])
    for i in range(7):
        df = pd.read_csv("all_mustard_csr{}.tsv".format(i+1), delimiter='\t', header=0, usecols=["sentence","utterance"])
        sentences = df['sentence']
        utterances = df['utterance']
        for i in range(len(utterances)):
            sent = sentences[i]
            each = utterances[i]
            each = each.replace('[','').replace(']','').replace(', ',',').replace("'",'')
            each = each.split(',')
            each = check_duplicates(each)  
            print('Appending new row to file...')
            tsv_writer.writerow([sent,each])

out_file.close()
        