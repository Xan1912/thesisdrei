'''
@All OS commands to run models for 20 iteraitons 
'''

import os
'''
    run BiLSTM
'''
# for i in range(20):
#     os.system("python3 Desktop/Thesis/bi_LSTM.py")

''' 
    run BERT 
'''

for i in range(20):
    os.system("python3 main.py 0.7 32 2e-5 1e-8 4 42")

''' 
    run DistillBERT 
'''

for i in range(20):
    os.system("python3 DISTmain.py 0.7 32 2e-5 1e-8 4 42")






