# from DISTmodel import DistBert
from k_encode import execute
from reasoner_trainer import prepare_data
from reasoner import train,test
# from DISTmodelJoint import DistBert # DISTBert for joint model
from DISTmodelReason import DistBert # DISTBert for joint model
from DISTeval import Eval
import argparse
import wandb
# from config import sweep_id
# from joint_model import JointModel
# from joint_model_2 import JointModel
from joint_model_K_Fold import JointModel
from DISTtokenizer import labels

'''Write the preprocessing file for MUSTARD'''
# from reasoner import execute_reasoner
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

# learning_rate = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]

# wandb.init(config=args)
# config = wandb.config

# for each in learning_rate:
#     # actvmodl = DistBert(args.train_set_size, args.batch_size, args.lr, args.eps, args.epochs, args.seed_val)
# actvmodl = DistBert(config)

#### for non-joint model ####

# pred = actvmodl.return_prediction_set()
# modl = actvmodl.train()
# wandb.agent(sweep_id, actvmodl.train)
# Eval(modl,pred)

# #### for joint model ####

# h_sts = actvmodl.train()

# val = actvmodl.return_validation_set()
# pred = actvmodl.return_prediction_set()

# with open('all_loudness.tsv') as f:
#     data = f.readlines()[1:]
#     s_sts = list()
#     for each in data: 
#         each = each.replace('[','').replace(']','').replace('\n','')
#         each = each.split(',')
#         x_each = []
#         for x in each:
#             x = float(x)
#             x_each.append(x)
#         s_sts.append(x_each)

# TEST_s_sts = s_sts[::-1]
# TEST_s_sts = TEST_s_sts[:167]
# TEST_s_sts = TEST_s_sts[::-1]

# # 482 42 167
# jointModel = JointModel(h_sts,s_sts[:482])
# jointModel.project()

# combined = jointModel.make_one_array()
# out = jointModel.combine()

# pred = jointModel.make_one_array_test(pred)
# test_pr_states = jointModel.project_test(TEST_s_sts)
# combined_test = jointModel.combine_test(pred, test_pr_states)

# # val = jointModel.make_one_array_test(val)

# # out_labels = labels.reverse()
# # out_labels = out_labels[:167]

# labels = labels[:482]
# inp = list(zip(out,labels))

# # jointModel.train(inp, val, pred)
# # debug = jointModel.train(inp, val, pred)

# # python3 -i DISTmain.py 0.7 32 1e-5 1e-8 2 42

# jointModel.train_and_evaluate_KFolds(inp)
# # jointModel.test_K_Folds(pred)
# jointModel.test_K_Folds(combined_test)


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

trainDataloader, validationDataloader, predictionDataloader = prepare_data(combined)

train(trainDataloader, validationDataloader)
x = test(predictionDataloader)







    
