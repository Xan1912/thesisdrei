import torch.nn as nn
import torch.nn.functional as F
import torch 
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

class Net(nn.Module):
    def __init__(self):
        # https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89
        super(Net,self).__init__()
        self.relu = nn.ReLU()
        self.layer_1 = nn.Linear(768,768)
        self.batchnorm1 = nn.LayerNorm(768)
        # insert LSTM layer
        self.lstm = nn.LSTM(768,768) # num_layers=2 : Two stacked LSTMs?
        self.layer_2 = nn.Linear(768, 64) # break this to several layers
        self.batchnorm2 = nn.LayerNorm(64)
        self.dropout = nn.Dropout(p=0.1)
        self.fc_1 = nn.Linear(64, 1) 
         
    def forward(self, x, i=None, h_lstm=None, epoch=None):

        len_x = len(x[0])
        print("#### X Shape: ", x.shape)
        x = self.relu(self.layer_1(x))
        x = self.batchnorm1(x)
        
        # h0 and c0 may need fixes
        if epoch == 0:
            h0 = torch.randn(1,len_x,768)
            c0 = torch.randn(1,len_x,768)
        else:
            h0 = h_lstm[i][0]
            c0 = h_lstm[i][1]

        x, (hn, cn) = self.lstm(x, (h0, c0))

        h_lstm[i] = (hn, cn)
        
        # I want the h0 and c0 to continue from the previous epochs
        print('###################################\n')
        print('#### Before Slice: ',x.shape)
        print('\n')
        x = x[:,-1]
        print('#### After Slice: ',x.shape)
        print('\n')
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.fc_1(x)

        return (x,h_lstm) 

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
            each = each.view(1,768)
            self.pr_states.append(each)

    def make_one_array(self):
        self.H_sts = []
        for each in self.h_sts:
            current = each 
            for i in current:
                self.H_sts.append(i)
        
        return self.H_sts

    def make_one_array_test(self,pred):
        all_pred = []
        for each in pred:
            current = each 
            for i in current:
                all_pred.append(i)
        
        return all_pred

    def combine(self):
        HS = list()
        for i in range(len(self.H_sts)):
            h = self.H_sts[i]
            s = self.pr_states[i]
            hs = torch.cat((h,s),0)
            HS.append(hs)

        return HS

    def train(self,inp, val_set, test_set): # change to train set
        net.train()
        self.inp = inp
        self.val_set = val_set
        
        trainloader = torch.utils.data.DataLoader(self.inp, batch_size = 32) # make experiments with different parameters here 
        valloader = torch.utils.data.DataLoader(self.val_set, batch_size = 32) # make experiments with different parameters here 

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        h_lstm = dict()

        for epoch in range(10):  # loop over the dataset multiple times

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
                print(i)
                outputs, h_lstm = net(inputs, i, h_lstm, epoch)
                print(outputs)
                loss = loss_fn(outputs, torch.unsqueeze(labels,1))
                acc = binary_acc(outputs, torch.unsqueeze(labels,1))
                
                loss.backward(retain_graph=True)
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_acc += acc.item()
                
                print('TRAINING STATS\n')

                print(f'Epoch {epoch+0:03}: | Loss: {epoch_loss/len(trainloader):.5f} | Acc: {epoch_acc/len(trainloader):.3f}')

                print('\n###################################')
            print('\nH_LSTM\n', h_lstm)

            """
            epoch_val_loss = 0
            epoch_val_acc = 0
            
            for j, data in enumerate(valloader, 0):
                
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.detach()
                labels = labels.float()
                with torch.no_grad():
                    outputs = net(inputs)
                
                    loss = loss_fn(outputs, torch.unsqueeze(labels,1))
                    acc = binary_acc(outputs, torch.unsqueeze(labels,1))
                  
                epoch_val_loss += loss.item()
                epoch_val_acc += acc.item()
                
                print('VALIDATION STATS\n')

                print(f'Epoch {epoch+0:03}: | Loss: {epoch_val_loss/len(valloader):.5f} | Acc: {epoch_val_acc/len(valloader):.3f}')

                print('\n###################################')
                """

        # self.PATH = './jnt_mod.pth'
        # torch.save(net.state_dict(), self.PATH)

        # return inputs

    # def test(self, test_set):

        self.test_set = test_set
        testloader = torch.utils.data.DataLoader(self.test_set, batch_size=16)
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
                y_test_pred, _ = net(X_batch)
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

        # return temp