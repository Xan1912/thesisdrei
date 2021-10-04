import requests
import nltk 
import time
from pathlib import Path
import csv

limitLeft = 119 # Although in theory some other code could be hitting the API, start with a presumption that we have all 500.

# from Stack Overflow
def request_rate_limited(request_function):
    def limit_rate(*args, **kwargs):
        global limitLeft
        if limitLeft < 120: # DEBUG LINE ONLY
            print('API calls remaining before making call: ' + str(limitLeft)) # DEBUG LINE ONLY
        if limitLeft < 10:
            print('API calls remaining approaching zero (below ten); sleeping for 2 seconds to refresh.')
            time.sleep(5)
        response = request_function(*args, **kwargs)
        #limitLeft = int(response.headers['X-Rate-Limit-Limit'])
        print('API calls remaining after making call: ' + str(limitLeft)) # DEBUG LINE ONLY
        return response
    return limit_rate

# Attach request_rate_limited() to "requests.get"     
requests.get = request_rate_limited(requests.get)

def get_surface_texts(word): 
    weight_ls = list()
    i = 0
    obj = requests.get('http://api.conceptnet.io/c/en/'+word).json()
    for i in range(len(obj['edges'])):
        weight_ls.append(obj['edges'][i]['weight'])
    avg_wght = calculate_avgweight(weight_ls)
    surfaceTexts=list()
    try:
        for i in range(len(obj['edges'])):
            weight = obj['edges'][i]['weight']
            if weight >= avg_wght:
                surface_text = obj['edges'][i]['surfaceText']
                if surface_text != None and 'translation' not in surface_text: 
                    surfaceTexts.append(surface_text.replace('[','').replace(']',''))
    except KeyError:
        print(obj)
    
    return surfaceTexts

def remove_palindrome_texts(string1,string2):
    pass
    
def calculate_avgweight(weight_ls):
    try: 
        return sum(weight_ls)/len(weight_ls)
    except ZeroDivisionError:
        print("Empty array!")
        
# main dataset folder 
data_folder = Path("datasets/"); path_to_IAC = data_folder / "MUSTARD/"; bert_input_txt_file = path_to_IAC / "bert-input.txt" ; 
with open(bert_input_txt_file, 'r') as f:
    sentences = f.readlines()[800:856]

'''
sentences = ["It's just a privilege to watch your mind at work.",
            "I don't think I'll be able to stop thinking about it.",
            "Since it's not bee season, you can have my epinephrine."]
'''

tagged_sentences = [(each,nltk.pos_tag(nltk.word_tokenize(each.strip('\n')))) for each in sentences]

i=0
tag_dict = dict()
for sentence in tagged_sentences:
    sent = sentence[0].strip('\n')
    tags = sentence[1]
    tag_dict[i] = (sent, set())
    for each in tags:
        if each[1] == 'NN' or each[1] == 'VB' or each [1] == 'JJ':
            tag_dict[i][1].add(each[0])
    i+=1

surface_text_dict = dict()

for key in tag_dict.keys():
    sent = tag_dict[key][0]
    val = tag_dict[key][1]
    surface_text_dict[key] = [sent, list()]
    for word in val:
        if word != '..':
            surface_texts = get_surface_texts(word)
        if surface_text_dict[key][1] == []:
            surface_text_dict[key][1] = surface_texts
        else:
            surface_text_dict[key][1] += surface_texts

with open('all_mustard_csr8.tsv', 'w') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['id','sentence', 'utterance'])
    for key,val in surface_text_dict.items():    
        __id = key 
        __sentence = val[0]
        __utterance = val[1]   
        print('Appending new row to file...')
        tsv_writer.writerow([__id,__sentence,__utterance])

out_file.close()
