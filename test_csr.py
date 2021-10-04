import requests
import json
import nltk
from nltk import word_tokenize

sentence = "Was she on her knees, taking lessons from you or was she skilled already?"
text = word_tokenize(sentence)
pos_tag_list = nltk.pos_tag(text)

####

#[('Was', 'NNP'), ('she', 'PRP'), ('on', 'IN'), ('her', 'PRP'), ('knees', 'NNS'), (',', ','), ('taking', 'VBG'), ('lessons', 'NNS'), 
#('from', 'IN'), ('you', 'PRP'), ('or', 'CC'), ('was', 'VBD'), ('she', 'PRP'), ('skilled', 'VBD'), ('already', 'RB'), ('?', '.')]

# she - PRP
# her - PRP
# you - PRP
# knees - NNS
# lessons - NNS
# skilled - VBD
# already - RB

####

lexicon = list()
for each in pos_tag_list:
    word = each[0]
    pos = each[1]

    if pos in ['PRP','NNS','VBD','RB']:
        lexicon.append(word)


data = requests.get('http://api.conceptnet.io/c/en/knees')
print(data.json())

