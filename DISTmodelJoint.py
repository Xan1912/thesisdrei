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
# import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
# logging info (later to be changed to WandB)
logging.basicConfig(level=logging.INFO)

class DistBert() :
    def __init__(self, train_set_size, batch_size, lr, eps, epochs, seed_val):
        self.train_set_size = train_set_size 
        self.batch_size = batch_size 
        self.lr = lr 
        self.eps = eps 
        self.epochs = epochs 
        self.seed_val = seed_val
        self.val_set_size = 0.2
        # Combine the training inputs into a TensorDataset.
        dataset = TensorDataset(input_ids, attention_masks, labels) 
        self.train_size = int(self.train_set_size * len(dataset))
        rest = len(dataset) - self.train_size
        self.val_size = int(self.val_set_size * rest)
        self.test_size = rest - self.val_size
        
        # Divide the dataset by randomly selecting samples.
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [self.train_size, self.val_size, self.test_size])
        print('{:>5,} training samples'.format(self.train_size))
        print('{:>5,} validation samples'.format(self.val_size))
        print('{:>5,} test samples'.format(self.test_size))

        # Create the DataLoaders for training and validation sets taking training samples in random order. 
        self.train_dataloader = DataLoader(
            self.train_dataset,                                  # training samples.
            sampler = RandomSampler(self.train_dataset),         # select batches randomly
            batch_size = self.batch_size,                        # trains with this batch size.
        )
        # For validation the order doesn't matter, so it will be read sequentially.
        self.validation_dataloader = DataLoader(
            self.val_dataset,                                    # validation samples.
            sampler = SequentialSampler(self.val_dataset),       # select batches sequentially.
            batch_size = self.batch_size                         # evaluate with this batch size.
        )

        prediction_sampler = SequentialSampler(self.test_dataset)
        self.prediction_dataloader = DataLoader(self.test_dataset, sampler=prediction_sampler, batch_size=self.batch_size)

        # Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',                              # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = 2,                                         # The number of output labels--2 for binary classification.   
            output_attentions = False,                              # Whether the model returns attentions weights.
            output_hidden_states = True,                           # Whether the model returns all hidden-states.
            return_dict=True
        )
        # Tell pytorch to run this model on the GPU.
        # model.cuda()
        # Adam Optmizer
        self.optimizer = AdamW(self.model.parameters(),
                        lr = self.lr, # args.learning_rate - default is 5e-5 (also use 2e-5)
                        eps = self.eps # args.adam_epsilon  - default is 1e-8
                        )
        # Total number of training steps is [number of batches] x [number of epochs]. 
        self.total_steps = len(self.train_dataloader) * self.epochs
        # Create the learning rate scheduler.
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                    num_warmup_steps = 0, 
                                                    num_training_steps = self.total_steps)                                            
        
        random.seed(self.seed_val)
        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)
        torch.cuda.manual_seed_all(self.seed_val)
        # Store a number of quantities such as training and validation loss, validation accuracy, and timings.
        self.training_stats = []
        # Measure the total training time for the whole run.
        self.total_t0 = time.time()

    def return_prediction_set(self):
        pred_sts = list()

        for batch in self.prediction_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch  
            # Telling the model not to compute or store gradients, saving memory and speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                res = self.model(b_input_ids, attention_mask=b_input_mask)
                # change BERT states later
                h_sts = res['hidden_states'][6] # take the last layer
                H = list(zip(h_sts,b_labels))
                pred_sts.append(H)
        
        return pred_sts

    def return_validation_set(self):
        val_sts = list()

        for batch in self.validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch  
            # Telling the model not to compute or store gradients, saving memory and speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                res = self.model(b_input_ids, attention_mask=b_input_mask)
                # change BERT states later
                h_sts = res['hidden_states'][6] # take the last layer
                H = list(zip(h_sts,b_labels))
                val_sts.append(H)
        
        return val_sts

    def train(self):
        for epoch_i in range(0, self.epochs):
            # Perform one full pass over the training set.
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')
            # Measure how long the training epoch takes.
            t0 = time.time()
            # Reset the total loss for this epoch.
            total_train_loss = 0
            # Put the model into training mode. 
            self.model.train()
            # For each batch of training data.
            all_h_states = list()
            for step, batch in enumerate(self.train_dataloader):
                # Progress update every 40 batches.
                if step % 60 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)
                    # Report progress.
                    print('Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.train_dataloader), elapsed))
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                # Clear any previously calculated gradients 
                self.model.zero_grad()        
                # Perform a forward pass (evaluate the model on this training batch).
                outputs = self.model(b_input_ids, 
                                    attention_mask=b_input_mask, 
                                    labels=b_labels)
                loss = outputs["loss"]
                logits = outputs["logits"]
                h_states = outputs["hidden_states"]
                all_h_states.append(h_states[6])
                # Perform a backward pass to calculate the gradients.
                loss.backward()
                total_train_loss += loss.item()
                # prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                # Update the learning rate.
                self.scheduler.step()
            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(self.train_dataloader)            
            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)
            print("")
            print("Average training loss: {0:.2f}".format(avg_train_loss))
            print("Training epcoh took: {:}".format(training_time))
        
        return all_h_states

        










































































































