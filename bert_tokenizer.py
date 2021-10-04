from transformers import BertTokenizer
# from parse_dataset import sentences, labels
from parse_dataset_csr import sentences, labels
import torch

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

################### Tokenize dataset ################################
# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []
# For every sentence...
for each in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        each,                                   # Sentence to encode.
                        add_special_tokens = True,              # Add '[CLS]' and '[SEP]'
                        max_length = 64,                        # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,           # Construct attn. masks.
                        return_tensors = 'pt',                  # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors. (fix the torch error in .pylintrc)
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels, dtype=torch.long)

'''
# Print sentence 0, now as a list of IDs.
print('Original: ', sentences[0])
print('Token IDs:', input_ids[0])
'''


'''
# Print the original sentence.
print('sanity check: print the first sentence with its token and token IDs')
print(' Original: ', sentences[0])
# Print the sentence split into tokens.
print('Tokenized: ', tokenizer.tokenize(sentences[0]))
# Print the sentence mapped to token ids.
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))

# Find the maximum length of a sentence 
max_len = 0
for each in sentences:
    TOKENIZED = tokenizer.tokenize(each)
    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(each, add_special_tokens=True)
    # Update the maximum sentence length.
    max_len = max(max_len, len(input_ids))

print('Input IDs:', input_ids)
print('Max sentence length: ', max_len)
'''
