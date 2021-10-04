import torch
from transformers import BertTokenizer, BertModel
import logging
import matplotlib.pyplot as plt

## -------------------------------------------------------------------------------------------------------------------
##   Stuff that BERT likes
## -------------------------------------------------------------------------------------------------------------------
##   A special token, [SEP], to mark the end of a sentence, or the separation between two sentences
##   A special token, [CLS], at the beginning of our text. This token is used for classification tasks, 
##   but BERT expects it no matter what your application is.
##   Tokens that conform with the fixed vocabulary used in BERT
##   The Token IDs for the tokens, from BERT’s tokenizer
##   Mask IDs to indicate which elements in the sequence are tokens and which are padding elements
##   Segment IDs used to distinguish different sentences
##   Positional Embeddings used to show token position within the sequence
## --------------------------------------------------------------------------------------------------------------------


logging.basicConfig(level=logging.INFO)

# load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "Here is the sentence I want embeddings for."
marked_text = "[CLS] " + text + " [SEP]"

# Tokenize our sentence with the BERT tokenizer.
tokenized_text = tokenizer.tokenize(marked_text)

# list BERT vocabularies 
# list(tokenizer.vocab.keys())[5000:5020]

# Define a new example sentence with multiple meanings of the word "bank"
## text = "After stealing money from the bank vault, the bank robber was seen " \
##       "fishing on the Mississippi river bank."

# Add the special tokens.
## marked_text = "[CLS] " + text + " [SEP]"

# Define a new example sentence with multiple meanings of the word "bank"
text = "After stealing money from the bank vault, the bank robber was seen " \
       "fishing on the Mississippi river bank."

# Add the special tokens.
marked_text = "[CLS] " + text + " [SEP] "

# Split the sentence into tokens.
tokenized_text = tokenizer.tokenize(marked_text)

# Map the token strings to their vocabulary indeces. the function is ------> convert_tokens_to_ids() 
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Display the words with their indeces.

# for tup in zip(tokenized_text, indexed_tokens):
#     print('{:<12} {:>6,}'.format(tup[0], tup[1]))

# Mark each of the 22 tokens as belonging to sentence "1".
segments_ids = [1] * len(tokenized_text)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])


# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

# Run the text through BERT, and collect all of the hidden states produced from all 12 layers. 

with torch.no_grad():

    outputs = model(tokens_tensor, segments_tensors)

    # Evaluating the model will return a different number of objects based on 
    # how it's  configured in the `from_pretrained` call earlier. In this case, 
    # becase we set `output_hidden_states = True`, the third item will be the 
    # hidden states from all layers. See the documentation for more details:
    # https://huggingface.co/transformers/model_doc/bert.html#bertmodel

    hidden_states = outputs[2]


print ("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
layer_i = 0

print ("Number of batches:", len(hidden_states[layer_i]))
batch_i = 0

print ("Number of tokens:", len(hidden_states[layer_i][batch_i]))
token_i = 0

print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))


######

# For the 5th token in our sentence, select its feature values from layer 5.
token_i = 5
layer_i = 5
vec = hidden_states[layer_i][batch_i][token_i]

# Plot the values as a histogram to show their distribution.
plt.figure(figsize=(10,10))
plt.hist(vec, bins=200)
# plt.show()

# `hidden_states` is a Python list.
print('      Type of hidden_states:         ', type(hidden_states))

# Each layer in the list is a torch tensor.
print('      Tensor shape for each layer:   ', hidden_states[0].size())


# Concatenate the tensors for all layers. We use `stack` here to
# create a new dimension in the tensor.
token_embeddings = torch.stack(hidden_states, dim=0)
print(token_embeddings.size())

# Remove dimension 1, the "batches".
token_embeddings = torch.squeeze(token_embeddings, dim=1)
print(token_embeddings.size())

# Swap dimensions 0 and 1.
token_embeddings = token_embeddings.permute(1,0,2)
print(token_embeddings.size())

# Stores the token vectors, with shape [22 x 3,072]
token_vecs_cat = []

# `token_embeddings` is a [22 x 12 x 768] tensor.

# For each token in the sentence...
for token in token_embeddings:
    
    # 'token' is a [12 x 768] tensor

    # Concatenate the vectors (that is, append them together) from the last four layers.
    # Each layer vector is 768 values, so `cat_vec` is length 3,072.
    cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
    
    # Use `cat_vec` to represent `token`.
    token_vecs_cat.append(cat_vec)

print ('Shape of concatenated vector is: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))

#### one can also make mean & sum operations in torch and 
#### the respective functions are mean() and cat() besides the cat() function
#### but that is obviously dependent on the applications



