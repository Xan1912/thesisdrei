# import libraries
import torch
import logging
# import torch specific libraries
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# import transformers specific libraries
from transformers import DistilBertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
# my own modules
from DISTtokenizer import input_ids,attention_masks,labels
from gpu_config import device
from helper import flat_accuracy, format_time
# import basic modules 
import time
import random
import numpy as np
import pandas as pd
# import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
# logging info (later to be changed to WandB)
logging.basicConfig(level=logging.INFO)

train_set_size = 0.7
batch_size = 32 
lr = 1e-5
eps = 1e-8 
epochs = 4 
seed_val = 42
val_set_size = 0.2
# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels) 
train_size = int(train_set_size * len(dataset))
rest = len(dataset) - train_size
val_size = int(val_set_size * rest)
test_size = rest - val_size
        
# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))
print('{:>5,} test samples'.format(test_size))

# Create the DataLoaders for training and validation sets taking training samples in random order. 
train_dataloader = DataLoader(
    train_dataset,                                  # training samples.
    sampler = RandomSampler(train_dataset),         # select batches randomly
    batch_size = batch_size,                        # trains with this batch size.
)
# For validation the order doesn't matter, so it will be read sequentially.
validation_dataloader = DataLoader(
    val_dataset,                                    # validation samples.
    sampler = SequentialSampler(val_dataset),       # select batches sequentially.
    batch_size = batch_size                         # evaluate with this batch size.
)

prediction_sampler = SequentialSampler(test_dataset)
prediction_dataloader = DataLoader(test_dataset, sampler=prediction_sampler, batch_size=batch_size)

# Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',                              # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2,                                         # The number of output labels--2 for binary classification.   
    output_attentions = False,                              # Whether the model returns attentions weights.
    output_hidden_states = False,                           # Whether the model returns all hidden-states.
    return_dict=False
)
# Tell pytorch to run this model on the GPU.
# model.cuda()
# Adam Optmizer
optimizer = AdamW(model.parameters(),
                lr = lr, # args.learning_rate - default is 5e-5 (also use 2e-5)
                eps = eps # args.adam_epsilon  - default is 1e-8
                )
# Total number of training steps is [number of batches] x [number of epochs]. 
total_steps = len(train_dataloader) * epochs
# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps)                                            
        
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
# Store a number of quantities such as training and validation loss, validation accuracy, and timings.
training_stats = []
# Measure the total training time for the whole run.
total_t0 = time.time()

for epoch_i in range(0, epochs):
    # Perform one full pass over the training set.
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    # Measure how long the training epoch takes.
    t0 = time.time()
    # Reset the total loss for this epoch.
    total_train_loss = 0
    # Put the model into training mode. 
    model.train()
    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):
        # Progress update every 40 batches.
        if step % 60 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            # Report progress.
            print('Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        # Clear any previously calculated gradients 
        model.zero_grad()        
        # Perform a forward pass (evaluate the model on this training batch).
        loss, logits = model(b_input_ids, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)
        total_train_loss += loss.item()
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        '''
        Update parameters and take a step using the computed gradient.
        The optimizer dictates the "update rule"--how the parameters are
        modified based on their gradients, the learning rate, etc.
        '''
        optimizer.step()
        # Update the learning rate.
        scheduler.step()
    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)
    print("")
    print("Average training loss: {0:.2f}".format(avg_train_loss))
    print("Training epcoh took: {:}".format(training_time))
           
    print("")
    print("Running Validation...")
    t0 = time.time()
    model.eval()
    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    # Evaluate data for one epoch
    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        with torch.no_grad():        
            (loss, logits) = model(b_input_ids,  
                                attention_mask=b_input_mask,
                                labels=b_labels)
        # Accumulate the validation loss.
        total_eval_loss += loss.item()
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)
    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))
    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
# summary training process 
pd.set_option('precision', 2)
# Create a DataFrame from our training statistics.
df_stats = pd.DataFrame(data=training_stats)
# Use the 'epoch' as the row index.
df_stats = df_stats.set_index('epoch')
# A hack to force the column headers to wrap.
# df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])
# Display the table.
print(df_stats)
# plot stuff 
sns.set(style='darkgrid')
# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)
# Plot the learning curve.
plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")
# Label the plot.
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks([1, 2])
plt.show()