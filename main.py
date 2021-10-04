from seqtoseqBERT import BertSeqToSeq
from eval_seqtoseqBERT import Eval
import argparse
import wandb

parser = argparse.ArgumentParser(description="Training parameters")
parser.add_argument("train_set_size", type=float)
parser.add_argument("batch_size", type=int) 
parser.add_argument("lr", type=float)
parser.add_argument("eps", type=float)
parser.add_argument("epochs", type=int)
parser.add_argument("seed_val", type=int)
args = parser.parse_args()

wandb.init(config=args)
config = wandb.config

actvmodl = BertSeqToSeq(config)
pred = actvmodl.return_prediction_set()
modl = actvmodl.train()
Eval(modl,pred)
