# import libraries
import torch
import logging
# import torch specific libraries
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# import transformers specific libraries
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
# my own modules
from bert_tokenizer import input_ids,attention_masks,labels
from gpu_config import device
from helper import flat_accuracy, format_time
# import basic modules 
import time
import random
import numpy as np
# import plotting libraries
import wandb

# logging info (later to be changed to WandB)
logging.basicConfig(level=logging.INFO)

class BertSeqToSeq():
    def __init__(self, config):
        self.train_set_size = config['train_set_size'] 
        self.batch_size = config['batch_size'] 
        self.lr = config['lr'] 
        self.eps = config['eps'] 
        self.epochs = config['epochs'] 
        self.seed_val = config['seed_val']
        self.val_set_size = 0.15
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
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",                                    # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = 2,                                         # The number of output labels--2 for binary classification.   
            output_attentions = False,                              # Whether the model returns attentions weights.
            output_hidden_states = False,                           # Whether the model returns all hidden-states.
            return_dict=False
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
        return self.prediction_dataloader

    def return_validation_set(self):
        return self.validation_dataloader

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
            # For each batch of training data...
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
                loss, logits = self.model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask, 
                                    labels=b_labels)
                total_train_loss += loss.item()
                # Perform a backward pass to calculate the gradients.
                loss.backward()
                '''
                Clip the norm of the gradients to 1.0.
                This is to help prevent the "exploding gradients" problem.
                '''
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                '''
                Update parameters and take a step using the computed gradient.
                The optimizer dictates the "update rule"--how the parameters are
                modified based on their gradients, the learning rate, etc.
                '''
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

            train_metrics = {"training_loss": avg_train_loss}
            wandb.log(train_metrics)

            self.val()

        return self.model

    def val(self):
        print("")
        print("Running Validation...")
        t0 = time.time()
        self.model.eval()
        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        # Evaluate data for one epoch
        for batch in self.validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            with torch.no_grad():        
                (loss, logits) = self.model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
            # Accumulate the validation loss.
            total_eval_loss += loss.item()
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)
        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(self.validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(self.validation_dataloader)
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))
        
        val_metrics = {"validation_accuracy": avg_val_accuracy,
               "validation_loss": avg_val_loss}
        wandb.log(val_metrics)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # Record all statistics from this epoch.
        # self.training_stats.append(
        #     {
        #         'epoch': epoch_i + 1,
        #         'Training Loss': self.avg_train_loss,
        #         'Valid. Loss': avg_val_loss,
        #         'Valid. Accur.': avg_val_accuracy,
        #         'Training Time': self.training_time,
        #         'Validation Time': validation_time
        #     }
        # )
    # print("Training complete!")
    # print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-self.total_t0)))
    # # summary training process 
    # pd.set_option('precision', 2)
    # # Create a DataFrame from our training statistics.
    # df_stats = pd.DataFrame(data=self.training_stats)
    # # Use the 'epoch' as the row index.
    # df_stats = df_stats.set_index('epoch')
    # # A hack to force the column headers to wrap.
    # # df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])
    # # Display the table.
    # print(df_stats)
    # # plot stuff 
    # sns.set(style='darkgrid')
    # # Increase the plot size and font size.
    # sns.set(font_scale=1.5)
    # plt.rcParams["figure.figsize"] = (12,6)
    # # Plot the learning curve.
    # plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    # plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")
    # # Label the plot.
    # plt.title("Training & Validation Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.xticks([1, 2])
    # plt.show()