from pathlib import Path
import pandas as pd
import csv
import json 

# main dataset folder 
datasets = Path("datasets/")

# path to the IAC folder
path_to_MUSTARD = datasets / "MUSTARD/sarcasm_data.json"
with open('all_mustard.tsv', 'w') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['id','utterance','speaker', 'context', 'contextspeakers', 'show', 'sarcasm'])    
    with open(path_to_MUSTARD) as our_json:
        __all = json.load(our_json)
        for key,val in __all.items():
            __id = key 
            __utterance = val["utterance"]
            __speaker = val["speaker"] 
            __context = '::'.join(val["context"])
            __contextspeakers = ',,'.join(val["context_speakers"])
            __show = val["show"]
            __sarcasm = val["sarcasm"]
            print('Appending new row to file...')
            tsv_writer.writerow([__id,__utterance,__speaker, __context, __contextspeakers, __show, __sarcasm])

out_file.close()

# with open('all_non_preprocessed_IAC.tsv', 'w') as out_file:
#     tsv_writer = csv.writer(out_file, delimiter='\t')
#     tsv_writer.writerow(['label','id','text'])
#     for each in files: s
#         dataFrame = pd.read_csv(each)
#         for key, val in dataFrame.iterrows():
#             print('Appending new row to file...')
#             tsv_writer.writerow([val[i] for i in range(len(val))])

# out_file.close()



    



