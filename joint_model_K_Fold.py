import torch.nn as nn
import torch.nn.functional as F
import torch 
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

# make sentence representation
def avg(inp):
    return torch.mean(inp, 0)

def K_Fold(k_folds=5):
    kfold = KFold(n_splits=k_folds, shuffle=True)
    return kfold

def reset_weights(m):
    '''
    Try resetting model weights to avoid weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

class Net(nn.Module):
    def __init__(self):
        # https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89
        super(Net,self).__init__()
        self.relu = nn.ReLU()
        self.layer_1 = nn.Linear(768,768)
        self.norm1 = nn.LayerNorm(768) 
        self.layer_2 = nn.Linear(768, 64)
        self.norm2 = nn.LayerNorm(64)
        self.dropout = nn.Dropout(p=0.1)
        self.fc_1 = nn.Linear(64, 1) 
         
    def forward(self, x):
        
        print("\n#### Input Shape: \n", x.shape)
        
        x = self.relu(self.layer_1(x))
        x = self.norm1(x)
        x = self.relu(self.layer_2(x))
        x = self.norm2(x)
        x = self.dropout(x)
        x = self.fc_1(x)

        return x

net = Net()

class JointModel:
    def __init__(self, h_sts, s_sts):
        self.h_sts = h_sts
        self.s_sts = s_sts
        self.pr_states=list()
     
    def project(self):
        for each in self.s_sts:
            self.l_layer = nn.Linear(len(each),768)
            each = self.l_layer(torch.Tensor(each))
            self.pr_states.append(each)

    def project_test(self, test_s_sts):
        self.test_s_sts = test_s_sts
        test_pr_states = []
        for each in self.test_s_sts:
            self.l_layer = nn.Linear(len(each),768)
            each = self.l_layer(torch.Tensor(each))
            test_pr_states.append(each)
        return test_pr_states

    def make_one_array(self):
        self.H_sts = []
        for each in self.h_sts:
            current = each 
            for i in current:
                i = avg(i)
                self.H_sts.append(i)
        
        return self.H_sts

    def make_one_array_test(self,pred):
        all_pred = []
        for each in pred:
            current = each 
            for i in current:
                x = avg(i[0])
                y = i[1]
                z = (x,y)
                all_pred.append((z))
        
        return all_pred

    def combine(self):
        HS = list()
        for i in range(len(self.H_sts)):
            h = self.H_sts[i]
            s = self.pr_states[i]
            hs = torch.mul(h,s)
            HS.append(hs)

        return HS

    def combine_test(self, pred, speech):
        HS = list()
        for i in range(len(pred)):
            h = pred[i][0]
            label = pred[i][1]
            s = speech[i]
            hs = torch.mul(h,s)
            final = (hs, label)
            HS.append(final)

        return HS

    def train(self,inp, val_set, test_set): # change to train set
        net.train()
        self.inp = inp
        self.val_set = val_set
        trainloader = torch.utils.data.DataLoader(self.inp, batch_size = 64) # make experiments with different parameters here 
        # valloader = torch.utils.data.DataLoader(self.val_set, batch_size = 64) # make experiments with different parameters here 

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(net.parameters(), lr=2e-5)

        for epoch in range(50):  # loop over the dataset multiple times

            epoch_loss = 0.0
            epoch_acc = 0.0
        
            for i, data in enumerate(trainloader, 0):
                
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.detach()
                labels = labels.float()
                # zero the parameter gradients
                optimizer.zero_grad() # maybe something fishy here and retain_graph = True
                
                # forward + backward + optimize
                outputs = net(inputs)
                
                loss = loss_fn(outputs, torch.unsqueeze(labels,1))
                acc = binary_acc(outputs, torch.unsqueeze(labels,1))
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_acc += acc.item()
                
                print('\nTRAINING STATS\n')

                print(f'Epoch {epoch+0:03}: | Loss: {epoch_loss/len(trainloader):.5f} | Acc: {epoch_acc/len(trainloader):.3f}')

                print('\n###################################')
            
            # epoch_val_loss = 0
            # epoch_val_acc = 0
            
            # for j, data in enumerate(valloader, 0):
                
            #     # get the inputs; data is a list of [inputs, labels]
            #     inputs, labels = data
            #     inputs = inputs.detach()
            #     labels = labels.float()
            #     with torch.no_grad():
            #         outputs = net(inputs)
                
            #         loss = loss_fn(outputs, torch.unsqueeze(labels,1))
            #         acc = binary_acc(outputs, torch.unsqueeze(labels,1))
                  
            #     epoch_val_loss += loss.item()
            #     epoch_val_acc += acc.item()
                
            #     print('VALIDATION STATS\n')

            #     print(f'Epoch {epoch+0:03}: | Loss: {epoch_val_loss/len(valloader):.5f} | Acc: {epoch_val_acc/len(valloader):.3f}')

            #     print('\n###################################')
        
        # self.PATH = './jnt_mod.pth'
        # torch.save(net.state_dict(), self.PATH)

        # return inputs

    # def test(self, test_set):

        self.test_set = test_set
        testloader = torch.utils.data.DataLoader(self.test_set, batch_size=4)
        # net.load_state_dict(torch.load(self.PATH))
        LABELS = list()
        y_pred_list = []
        PRED = list()
        net.eval()
        temp=[]
        print('Model in testing:')
        with torch.no_grad():
            for i,X_batch in enumerate(testloader):
                # X_batch = X_batch.to(device)
                X_batch, labels = X_batch
                y_test_pred = net(X_batch)
                temp.append(y_test_pred)
                y_test_pred = torch.sigmoid(y_test_pred)
                y_pred_tag = torch.round(y_test_pred)
                y_pred_list.append(y_pred_tag.cpu().numpy())
                LABELS += labels.tolist()

        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        
        for each in y_pred_list:
            for i in each:
                PRED.append(i)

        print('labels', LABELS)
        print('pred', PRED)
        print(classification_report(LABELS, PRED))

        return temp

    def train_and_evaluate_KFolds(self, inp): # change to train set
        self.inp = inp
        kfold = K_Fold()
        loss_fn = nn.BCEWithLogitsLoss()

        for fold, (train_ids, test_ids) in enumerate(kfold.split(self.inp)):
            print(f'FOLD {fold}')
            print('--------------------------------')

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            trainloader = torch.utils.data.DataLoader(self.inp, batch_size=64, sampler=train_subsampler)
            print('TRAINLOADER SIZE:  ',len(trainloader))
            
            testloader = torch.utils.data.DataLoader(self.inp, batch_size=4, sampler=test_subsampler)
            print('TESTLOADER SIZE:  ',len(testloader))
            
            net.train()
            # net.apply(reset_weights)
            optimizer = optim.Adam(net.parameters(), lr=0.1)
            
            for epoch in range(30):  # loop over the dataset multiple times
                
                print(f'Starting epoch {epoch+1}')
                epoch_loss = 0.0
                epoch_acc = 0.0
        
                for i, data in enumerate(trainloader, 0):
                
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    inputs = inputs.detach()
                    labels = labels.float()
                    # zero the parameter gradients
                    optimizer.zero_grad() # maybe something fishy here and retain_graph = True
                
                    # forward + backward + optimize
                    outputs = net(inputs)
                
                    loss = loss_fn(outputs, torch.unsqueeze(labels,1))
                    acc = binary_acc(outputs, torch.unsqueeze(labels,1))
                
                    loss.backward()
                    optimizer.step()
                
                    epoch_loss += loss.item()
                    epoch_acc += acc.item()
                
                    print('\nTRAINING STATS\n')

                    print(f'Epoch {epoch+0:03}: | Loss: {epoch_loss/len(trainloader):.5f} | Acc: {epoch_acc/len(trainloader):.3f}')

                    print('\n###################################')

            print('Training process has finished. Saving trained model.')
            print('Starting testing')
            
            # Saving the model
            self.PATH = f'./joint_model_fold.pth'
            torch.save(net.state_dict(), self.PATH)
            
            # Evaluation for this fold
            net.eval()
            
            y_pred_list = []
            temp=[]

            LABELS = list()
            PRED = list()
            
            print('Model in testing:')
            
            with torch.no_grad():
                for i,X_batch in enumerate(testloader):
                    # X_batch = X_batch.to(device)
                    X_batch, labels = X_batch
                    y_test_pred = net(X_batch)
                    temp.append(y_test_pred)
                    y_test_pred = torch.sigmoid(y_test_pred)
                    y_pred_tag = torch.round(y_test_pred)
                    y_pred_list.append(y_pred_tag.cpu().numpy())
                    LABELS += labels.tolist()

            y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        
            for each in y_pred_list:
                print('EACH IN FINAL PREDICTION LIST:  ',each)
                if isinstance(each,float):
                    PRED.append(i)
                else:
                    for i in each:
                        PRED.append(i)

            print('labels', LABELS)
            print('pred', PRED)
            print(classification_report(LABELS, PRED))

    # UNSEEN DATA TESTBED
    def test_K_Folds(self,test_set):
        print('TESTING ON UNSEEN DATA: ')
        self.test_set = test_set
        TESTloader = torch.utils.data.DataLoader(self.test_set, batch_size=4)
        net.load_state_dict(torch.load(self.PATH))
        LABELS = list()
        y_pred_list = []
        PRED = list()
        net.eval()
        temp=[]
        print('Model in testing:')
        with torch.no_grad():
            for i,X_batch in enumerate(TESTloader):
                # X_batch = X_batch.to(device)
                X_batch, labels = X_batch
                y_test_pred = net(X_batch)
                temp.append(y_test_pred)
                y_test_pred = torch.sigmoid(y_test_pred)
                y_pred_tag = torch.round(y_test_pred)
                y_pred_list.append(y_pred_tag.cpu().numpy())
                LABELS += labels.tolist()

        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        
        for each in y_pred_list:
            print('EACH IN FINAL PREDICTION LIST:  ',each)
            if isinstance(each,float):
                PRED.append(i)
            else:
                for i in each:
                    PRED.append(i)

        print('labels', LABELS)
        print('pred', PRED)
        print(classification_report(LABELS, PRED))