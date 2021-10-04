import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
import torch.optim as optim
from sklearn.metrics import classification_report
import wandb

class ScaledDotProductAttention(nn.Module):
    # Scaled Dot-Product Attention
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class Multi_Head_Attention(nn.Module):
    def __init__(self, d_model, n_head=4, d_k=64, d_v=64, dropout=0.1):
        super(Multi_Head_Attention,self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.d_model = d_model

        self.q_proj = nn.Linear(768,64)

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm([64,64], eps=1e-6)
        self.ff = nn.Linear(64,1)
        self.nf = 64
        w = torch.empty(64, 64)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(64))

        self.layer_norm3 = nn.LayerNorm([64, 64], eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        # sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        sz_b, len_q, len_k, len_v = 64, len(q), len(k), len(v)
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.q_proj(q)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        q, attn = self.attention(q, k, v, mask=mask)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q = self.layer_norm(q)
        q = q.reshape(q.size(1),sz_b,self.d_model)
        # normalization layer 1
        q = self.layer_norm2(q)
        # # conv1D
        size_out = q.size()[:-1] + (64,)
        q = torch.addmm(self.bias, q.view(-1, q.size(-1)), self.weight)
        q = q.view(*size_out)
        # feed forward layer 
        q = self.ff(q)
        q = q.reshape(q.size(0),64)
        q = self.ff(q) # May have to add a softmax here
        return q

'''
    Helper functions
'''

def K_Fold(k_folds=5):
    kfold = KFold(n_splits=k_folds, shuffle=True)
    return kfold

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

def create_separate_tensors(trainloader):
    INP = [item[0] for item in trainloader]
    RSN = [item[1] for item in trainloader]
    LBL = [item[2] for item in trainloader]
    result = [torch.stack(INP), torch.stack(RSN), torch.stack(LBL)]
    return result

'''
    Training function
'''

attn = Multi_Head_Attention(64)
PATH = f'./DIST_rsnr.pth'

def train(trainloader, valloader, batch_size=32): # change to train set
    loss_fn = nn.BCEWithLogitsLoss()
    attn.train()
    print('TRAINLOADER SIZE:  ',len(trainloader))
    # net.apply(reset_weights)
    optimizer = optim.Adam(attn.parameters(), lr=0.001)
    loader = create_separate_tensors(trainloader)
    v_loader = create_separate_tensors(valloader)
    inputs, knwldg, labels = loader 
    v_inputs, v_knwldg, v_labels = v_loader           
    for epoch in range(30):  # loop over the dataset multiple times
        print(f'Starting epoch {epoch+1}')
        epoch_loss = 0.0
        epoch_acc = 0.0    
        for i in range(0,len(inputs),batch_size):
            input_batch = inputs[i:i+batch_size]
            knwldg_batch = knwldg[i:i+batch_size]
            label_batch = labels[i:i+batch_size]
            q = input_batch.detach()
            k = knwldg_batch.detach()
            v = knwldg_batch.detach()
            label_batch = label_batch.float()
            # zero the parameter gradients
            optimizer.zero_grad() # maybe retain_graph = True
            outputs = attn(q,k,v)
            print("SHAPE OF OUTPUT: ")
            print(outputs.shape)
            loss = loss_fn(outputs, torch.unsqueeze(label_batch,1))
            acc = binary_acc(outputs, torch.unsqueeze(label_batch,1))
                
            loss.backward(retain_graph = True)
            optimizer.step()
                
            epoch_loss += loss.item()
            epoch_acc += acc.item()
                
            print('\nTRAINING STATS\n')
            print(f'Epoch {epoch+0:03}: | Loss: {epoch_loss/len(trainloader):.5f} | Acc: {epoch_acc/len(trainloader):.3f}')
            print('\n###################################')
            
        ######################################################## VALIDATION #######################################################################
            
        print("")
        print("Running Validation...")
        attn.eval()
        v_epoch_loss = 0.0
        v_epoch_acc = 0.0
        with torch.no_grad():
            for i in range(0,len(v_inputs),batch_size):
                v_input_batch = v_inputs[i:i+batch_size]
                v_knwldg_batch = v_knwldg[i:i+batch_size]
                v_label_batch = v_labels[i:i+batch_size]
                _q = v_input_batch.detach()
                _k = v_knwldg_batch.detach()
                _v = v_knwldg_batch.detach()
                v_label_batch = v_label_batch.float()
                # zero the parameter gradients
                v_outputs = attn(_q,_k,_v)
                print("SHAPE OF OUTPUT: ")
                print(outputs.shape)
                v_loss = loss_fn(v_outputs, torch.unsqueeze(v_label_batch,1))
                v_acc = binary_acc(v_outputs, torch.unsqueeze(v_label_batch,1))
                v_epoch_loss += v_loss.item()
                v_epoch_acc += v_acc.item()
                print('\nVALIDATION STATS\n')
                print(f'Epoch {epoch+0:03}: | Loss: {v_epoch_loss/len(valloader):.5f} | Acc: {v_epoch_acc/len(valloader):.3f}')
                print('\n###################################')    
    
    # Saving the model
    torch.save(attn.state_dict(), PATH)
    
def test(testloader, batch_size=32):
    attn.load_state_dict(torch.load(PATH))
    attn.eval()
    print('TESTLOADER SIZE:  ',len(testloader))
    # net.apply(reset_weights)
    t_loader = create_separate_tensors(testloader)
    t_inputs, t_knwldg, t_labels = t_loader 
    y_pred_list = list(); temp=list(); LABELS = list(); PRED = list()
    print('Model in testing:')
    with torch.no_grad():
        for i in range(0,len(t_inputs),batch_size):
            t_input_batch = t_inputs[i:i+batch_size]
            t_knwldg_batch = t_knwldg[i:i+batch_size]
            t_label_batch = t_labels[i:i+batch_size]
            __q = t_input_batch.detach()
            __k = t_knwldg_batch.detach()
            __v = t_knwldg_batch.detach()
            t_label_batch = t_label_batch.float()
            print('t:\n',t_label_batch)
            # zero the parameter gradients
            y_test_pred = attn(__q,__k,__v)
            temp.append(y_test_pred)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())
            LABELS += t_label_batch.tolist()

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    for each in y_pred_list:
        if isinstance(each,float):
            PRED.append(i)
        else:
            for i in each:
                PRED.append(i)

    print('labels', LABELS)
    print('pred', PRED)
    print(classification_report(LABELS, PRED))


