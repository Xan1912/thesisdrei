from transformers import DistilBertTokenizer
# from parse_dataset import sentences, labels
from parse_dataset_csr import sentences, labels
# from parse_dataset_iac import sentences, labels
import torch

# Load the DistillBERT tokenizer.
print('Loading DistillBERT tokenizer...')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

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
                        max_length = 64,                       # Pad & truncate all sentences.
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




