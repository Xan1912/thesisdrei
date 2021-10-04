import pandas as pd
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset, random_split
import torch 
import numpy as np
# import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
# import precision
from sklearn.metrics import classification_report
# import gpu_config
from gpu_config import device

# from Stack Ovf: cite later if used 
def perf_measure(y_true, y_pred):
  TP = 0;FP = 0;TN = 0;FN = 0
  for i in range(len(y_pred)): 
    if y_true[i]==y_pred[i]==1:
      TP += 1
    if y_pred[i]==1 and y_true[i]!=y_pred[i]:
      FP += 1
    if y_true[i]==y_pred[i]==0:
      TN += 1
    if y_pred[i]==0 and y_true[i]!=y_pred[i]:
      FN += 1
  return (TP, FP, TN, FN)

def precision(TP, FP):
  return float(TP/(TP+FP))

def recall(TP, FN):
  return float(TP/(TP+FN))

def fscore(pr,re):
  return float((2*pr*re)/(pr+re))

class Eval():
  def __init__(self, model, prediction_dataloader):
    # Put model in evaluation mode
    self.model = model
    self.model.eval()
    
    predictions , true_labels = [], []
    # Predict 
    for batch in prediction_dataloader:
      # Add batch to GPU
      batch = tuple(t.to(device) for t in batch)
      # Unpack the inputs from our dataloader
      b_input_ids, b_input_mask, b_labels = batch
      # Telling the model not to compute or store gradients, saving memory and speeding up prediction
      with torch.no_grad():
          # Forward pass, calculate logit predictions
          outputs = self.model(b_input_ids, attention_mask=b_input_mask)
      logits = outputs[0]
      # Move logits and labels to CPU
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()
      # Store predictions and true labels
      predictions.append(logits)
      true_labels.append(label_ids)

    print(' DONE.')
    print('******************************* Classification report: *******************************\n')

    # output_array_size = len(predictions)*len(predictions[0])
    y_true = list() # these have to be changed 
    y_pred = list()
    for i in range(len(true_labels)):
      pred_labels_i = np.argmax(predictions[i], axis=1)
      y_true += true_labels[i].tolist()
      y_pred += pred_labels_i.tolist()
    TP, FP, TN, FN = perf_measure(y_true, y_pred)
    pre = precision(TP,FP)
    rec = recall(TP, FN)
    fsc = fscore(pre,rec)

    print('Report via manual evaluation:\n')
    print('Precision: ', pre)
    print('Recall: ', rec)
    print('Fscore: ', fsc)
    print('Report via SKlearn:\n')
    print(classification_report(y_true, y_pred))























































'''
    # # main dataset folder ; path to the IAC folder ; define IAC path
    # data_folder = Path("datasets/") ; path_to_IAC = data_folder / "IAC/sarcasm_v2/" ; IAC_GEN = path_to_IAC / "GEN-sarc-notsarc.csv"
    # # Load the BERT tokenizer.
    # print('Loading DistillBERT tokenizer...')
    # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    # # Load the dataset into a pandas dataframe.
    # df = pd.read_csv(IAC_GEN, header=0, usecols=["label","text"])
    # # change data to boolean
    # df['label'] = df['label'].apply(lambda x: True if x == 'sarc' else False)
    # # Report the number of sentences.
    # print('Number of test sentences: {:,}\n'.format(df.shape[0]))
    # sentences = df.text.values
    # labels = df.label.values
    # # Tokenize all of the sentences and map the tokens to thier word IDs.
    # input_ids = []
    # attention_masks = []
    # # For every sentence...
    # for sent in sentences:
    #     # `encode_plus` will:
    #     #   (1) Tokenize the sentence.
    #     #   (2) Prepend the `[CLS]` token to the start.
    #     #   (3) Append the `[SEP]` token to the end.
    #     #   (4) Map tokens to their IDs.
    #     #   (5) Pad or truncate the sentence to `max_length`
    #     #   (6) Create attention masks for [PAD] tokens.
    #     encoded_dict = tokenizer.encode_plus(
    #                         sent,                           # Sentence to encode.
    #                         add_special_tokens = True,      # Add '[CLS]' and '[SEP]'
    #                         max_length = 64,                # Pad & truncate all sentences.
    #                         pad_to_max_length = True,
    #                         return_attention_mask = True,   # Construct attn. masks.
    #                         return_tensors = 'pt',          # Return pytorch tensors.
    #                   )
    #     # Add the encoded sentence to the list.    
    #     input_ids.append(encoded_dict['input_ids'])
    #     # And its attention mask (simply differentiates padding from non-padding).
    #     attention_masks.append(encoded_dict['attention_mask'])

    # # Convert the lists into tensors.
    # input_ids = torch.cat(input_ids, dim=0)
    # attention_masks = torch.cat(attention_masks, dim=0)
    # labels = torch.tensor(labels, dtype=torch.long)
    # # Set the batch size.  
    # batch_size = 32  
    # # Create the DataLoader.
    # prediction_data = TensorDataset(input_ids, attention_masks, labels)
    # test_set_size = 0.9
    # test_size = int(test_set_size * len(prediction_data))
    # rest = len(prediction_data) - test_size
    # # Divide the dataset by randomly selecting samples.
    # prediction_data, rest_unusued = random_split(prediction_data, [test_size, rest])
    # prediction_sampler = SequentialSampler(prediction_data)
    # prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    # Prediction on test set
    # print('Predicting labels for {:,} test sentences...'.format(test_size))
    '''

'''
# print('Positive samples: %d of %d (%.2f%%)' % (df.label.sum(), len(df.label), (df.label.sum() / len(df.label) * 100.0)))
from sklearn.metrics import matthews_corrcoef

matthews_set = []

# Evaluate each test batch using Matthew's correlation coefficient
print('Calculating Matthews Corr. Coef. for each batch...')

# For each input batch...
for i in range(len(true_labels)):
  
  # The predictions for this batch are a 2-column ndarray (one column for "0" 
  # and one column for "1"). Pick the label with the highest value and turn this
  # in to a list of 0s and 1s.
  pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
  
  # Calculate and store the coef for this batch.  
  matthews = matthews_corrcoef(true_labels[i], pred_labels_i)                
  matthews_set.append(matthews)

# Create a barplot showing the MCC score for each batch of test samples.
ax = sns.barplot(x=list(range(len(matthews_set))), y=matthews_set, ci=None)

plt.title('MCC Score per Batch')
plt.ylabel('MCC Score (-1 to +1)')
plt.xlabel('Batch #')

plt.show()

# Combine the results across all batches. 
flat_predictions = np.concatenate(predictions, axis=0)

# For each sample, pick the label (0 or 1) with the higher score.
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

# Combine the correct labels for each batch into a single list.
flat_true_labels = np.concatenate(true_labels, axis=0)

# Calculate the MCC
mcc = matthews_corrcoef(flat_true_labels, flat_predictions)

print('Total MCC: %.3f' % mcc)
'''