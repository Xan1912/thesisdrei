import pandas as pd
import csv
import os

directory = 'all_wav_ftrs/'
allFiles=list()
for filename in os.listdir(directory):
    if filename.endswith(".csv"): 
        filepath = os.path.join(directory, filename)
        allFiles.append(filepath)

with open('all_loudness.tsv', 'w') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['loudness'])
    for each in allFiles:
        df = pd.read_csv(each, delimiter=';', header=0, usecols=["pcm_loudness_sma"])
        loudlist = df["pcm_loudness_sma"].tolist()
        tsv_writer.writerow([loudlist])

out_file.close()