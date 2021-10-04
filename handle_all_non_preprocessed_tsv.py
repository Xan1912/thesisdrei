import pandas as pd
import csv

def handle_sentence(sentence):

    sentence_ls = sentence.split(' ')
    
    sentence_ls = remove_hyphen_words(sentence_ls)
    sentence_ls = remove_long_upper_case_words(sentence_ls)
    sentence_ls = remove_atsymbols(sentence_ls)
    sentence_ls = remove_http(sentence_ls)
    sentence_ls = remove_dots_and_undscrs(sentence_ls)
    sentence_ls = remove_weirdstuff_in_words(sentence_ls)
    
    return (' ').join(sentence_ls)


# ------------------ helper functions begin ------------------ #


# TO DO:: 
# Bridge the gaps in the non preprocessed file, Add necessary regexes

def remove_hyphen_words(sentence):
    return [each for each in sentence if each != '' and each[0]!='-']

def remove_long_upper_case_words(sentence):
    return [each for each in sentence if len(each) < 15]

def remove_atsymbols(sentence):
    return [each for each in sentence if '@' not in each]

def remove_http(sentence):
    return [each for each in sentence if 'http' or 'www' not in each]

def remove_dots_and_undscrs(sentence):
    # modify to write regex to remove dots
    return [each.replace('...',' ') for each in sentence if '...' or '___________' or '<' or '>' or '[' or ']' not in each]
    
def remove_weirdstuff_in_words(sentence):
    # modify to write XXXX in regex
    return [each for each in sentence if 'emoticonX' or 'emoticon_x' or '#' or 'XXX' or 'xxx' or '*' or ':x' or '&#' or ':p' not in each]

# ------------------ helper functions end ------------------ #



# ------------------ make the file ------------------ #

all_non_preprocessed_IAC = "all_non_preprocessed_IAC.tsv"

with open('all_preprocessed_IAC.tsv', 'w') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['label','id','text'])
    dataFrame = pd.read_csv(all_non_preprocessed_IAC, delimiter='\t')
    for key, val in dataFrame.iterrows():
        print('Appending new row to file...')
        val[2] = handle_sentence(val[2])
        tsv_writer.writerow([val[i] for i in range(len(val))])

out_file.close()

# ------------------ end file composition section ------------------ #



# ------------------ Begin test sentence block ------------------ #

ts = handle_sentence('BWAHAHAHAHHAAH! AMENDMENTS.............ARE PART OF THE CONSTITUTION.')
print(ts)

# ------------------ End test sentence block ------------------ #