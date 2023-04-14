import pickle
import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from clstp.utils import data_divide, search_track_pos

class SGAN_encoder(nn.Module):
    def __init__(self, trial=0):
        super(SGAN_encoder,self).__init__()
        
        embdding_size = 64
        hidden_size = 256

        self.embdding = nn.Linear(in_features=2,out_features=embdding_size)
        self.lstm = nn.LSTM(
            input_size = embdding_size,
            hidden_size = hidden_size,
            num_layers = 1,
            # batch_first = True, # (batch, seq, feature)
            dropout = 0.1)

        
    
    def forward(self,input_batch):
        x = self.embdding(input_batch)
        out = self.lstm(x)
        return out

class SGAN_dataset(Dataset):
    def __init__(self,item_idx) -> None:
        super().__init__()
        set_folder_path = './data/set/'
        history_len = 8

        meta_info = np.load('./data/meta/meta_info.npy')
        with open('./data/meta/data_name_list.pkl','rb') as path:
            data_name_list = pickle.load(path)
            path.close()
        
        set_file = []
        for i in range(len(data_name_list)):
            data_name = data_name_list[i]
            with open("".join([set_folder_path,data_name,'.pkl']),'rb') as path:
                file = pickle.load(path)
                path.close()
            set_file.append(file)
        
        self.item = []
        for index in item_idx:
            meta_item = meta_info[index]
            track = search_track_pos(meta_item,set_file[meta_item[0]])

            track_x = track[:history_len].transpose()
            track_y = track[history_len:].transpose()
            self.item.append((track_x,track_y,meta_item[0].item()))
    
    def __getitem__(self, index):
        return self.item[index]
    
    def __len__(self):
        return len(self.item)

def SGAN_data_loader(train_item_idx=None,valid_item_idx=None,test_item_idx=None,batch_size=1):
    '''
    output: train_loader:(pid_history_seq,pid_future_seq)
    '''
    if train_item_idx is not None:
        train_set = SGAN_dataset(train_item_idx)
        valid_set = SGAN_dataset(valid_item_idx)
        train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=4,drop_last=True)
        valid_loader = DataLoader(valid_set,batch_size=1,shuffle=False,num_workers=4,drop_last=True)
        return train_loader,valid_loader
    else:
        test_set = SGAN_dataset(test_item_idx)
        test_loader = DataLoader(test_set,batch_size=1,shuffle=False,num_workers=4,drop_last=True)
        return test_loader

def SGAN_obj(trial):

    # para input
    data_div_method = 'CV'

    # hyperparameters selection with optuna
    optimizer_name = trial.suggest_categorical("optimizer", ["RMSprop", "SGD", "Adam"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 8, 256,step=8)
    EPOCHS = trial.suggest_int("EPOCHS",5,30)

    # [train_set,validation_set,test_set]
    train_valid_array = np.load('./data/meta/train_valid.npy')
    data_div_para = trial.suggest_categorical("Cross Validation k", [5,10,20])

    train_validation_idx = data_divide(train_valid_array,para=data_div_para)

    net = SGAN_encoder(trial)
    optimizer = getattr(torch.optim, optimizer_name)(net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    if data_div_method=='CV':
        optuna_error = torch.tensor([])
        for CV_i in range(data_div_para):
            train_item_idx = train_validation_idx[CV_i][0]
            valid_item_idx = train_validation_idx[CV_i][1]
            train_loader,valid_loader = SGAN_data_loader(train_item_idx=train_item_idx,
                                                              valid_item_idx=valid_item_idx,
                                                              batch_size=batch_size)

            valid_error = torch.tensor([])
            for _ in range(EPOCHS):
                # train
                net = train(net=net,train_loader=train_loader,criterion=criterion,
                      optimizer=optimizer,batch_size=batch_size)
                # validation
                epoch_error,_ = valid(net,valid_loader,criterion)

                valid_error = torch.concat((valid_error,epoch_error))
            valid_error = torch.tensor([valid_error.mean()])

            # drop trail if error is not acceptable
            trial.report(valid_error.item(),CV_i)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            else:
                optuna_error = torch.concat((valid_error,optuna_error))

        optuna_error = optuna_error.mean()

    torch.save(net.state_dict(),'./model/LinearDMS/trial/trial_{}.model'.format(trial.number))
    return optuna_error

def train(net,train_loader,criterion,optimizer,batch_size):
    net.train()
    for _,(track_x,track_y,_) in enumerate(train_loader):
        optimizer.zero_grad()
        for batch_item in range(batch_size):
            track_input = track_x[batch_item]
            track_target = track_y[batch_item]

            out = net(track_input)
            loss = criterion(track_target[:,-1],out[:,-1])
            loss.backward()
        optimizer.step()    
    return net

def valid(net,valid_loader,criterion):
    error = torch.tensor([])
    net.eval()
    with torch.no_grad():
        for _,(track_x,track_y,_) in enumerate(valid_loader):
            out = net(track_x)
            loss = criterion(track_y[:,-1],out[:,-1])
            loss_tensor = torch.tensor([loss.item()])
            error = torch.concat((loss_tensor,error))
        epoch_error = error.mean()
        epoch_error_std = error.std()
    
    return torch.tensor([epoch_error]),epoch_error_std

def SGAN_test(net,test_loader,criterion):
    test_length = len(test_loader)
    error = torch.zeros(test_length,2)
    with torch.no_grad():
        for item_idx,(track_x,track_y,track_set_code) in enumerate(test_loader):
            out = net(track_x)
            loss = criterion(track_y[:,-1],out[:,-1])
            error[item_idx,0] = track_set_code
            error[item_idx,1] = loss

    set_code,set_count = torch.unique(error[:,0],return_counts=True)

    set_code_num = set_code.size()[0]
    table = torch.zeros(set_code_num+1,4)
    for i in range(set_code_num):
        SetCode = set_code[i]
        table[i,0] = SetCode
        table[i,1] = error[torch.nonzero(error[:,0]==SetCode),1].mean()
        table[i,2] = error[torch.nonzero(error[:,0]==SetCode),1].std()
        table[i,3] = set_count[i]
    
    table[set_code_num,0] = -1
    table[set_code_num,1] = error[:,1].mean()
    table[set_code_num,2] = error[:,1].std()

    return table