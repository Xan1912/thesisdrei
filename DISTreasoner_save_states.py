# from DISTmodel import DistBert
from k_encode import execute
from reasoner_trainer import prepare_data
from reasoner import train,test
from DISTmodelReason import DistBert 
import argparse
from DISTtokenizer import labels

'''
Optimized Params:
learning_rate = 1e-5
batch_size = 32
warmup = 600
max_seq_length = 128
num_train_epochs = 3.0

python3 -i DISTmain.py 0.7 32 1e-5 1e-8 4 42
'''

parser = argparse.ArgumentParser(description="Training parameters")
parser.add_argument("train_set_size", type=float)
parser.add_argument("batch_size", type=int)
parser.add_argument("lr", type=float)
parser.add_argument("eps", type=float)
parser.add_argument("epochs", type=int)
parser.add_argument("seed_val", type=int)
args = parser.parse_args()
actvmodl = DistBert(args.train_set_size, args.batch_size, args.lr, args.eps, args.epochs, args.seed_val)

##### with reasoning model ####

h_sts = actvmodl.train()

def make_one_array(h_sts):
    H_sts = []
    for each in h_sts:
        current = each 
        for i in current:
            H_sts.append(i)    
    return H_sts

h = make_one_array(h_sts)
k = execute()
l = labels

def combine(h, k, l):
    states= list()
    for i in range(len(h)):
        hs = h[i]
        kn = k[i]
        lb = l[i]
        hs = (hs, kn, lb)
        states.append(hs)

    return states

combined = combine(h,k,l)
with open('save_reasoner_states.txt','w') as f:
    f.write(str(combined))

# trainDataloader, validationDataloader, predictionDataloader = prepare_data(combined)

# train(trainDataloader, validationDataloader)
# x = test(predictionDataloader)







    
