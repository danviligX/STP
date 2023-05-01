import pickle
import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from clstp.utils import data_divide, make_mlp, search_group_track_pos

class SGAN_encoder(nn.Module):
    '''
    single core
    input:
        history_track:
    output:
        hidden_state: encoded sequences, size=[1,batch,hidden_size]
        cell_state: if use LSTM, size=[1,batch,hidden_size]
    '''
    def __init__(self, trial=0):
        super(SGAN_encoder,self).__init__()
        
        self.embadding_size = 64
        self.hidden_size = 256
        self.ifgru = 1

        self.embadding = nn.Linear(in_features=2,out_features=self.embadding_size)

        if self.ifgru:
            self.rnn = nn.GRU(input_size = self.embadding_size,hidden_size = self.hidden_size,num_layers = 1)
        else:
            self.rnn = nn.LSTM(input_size = self.embadding_size,hidden_size = self.hidden_size,num_layers = 1)

    def encode(self,input_batch):
        x = self.embadding(input_batch)

        if self.ifgru:
            _,state_tuple = self.rnn(x)
            return state_tuple
        else:
            _,state_tuple = self.rnn(x)
            return state_tuple

    def forward(self,group_track):
        HS = []
        if self.ifgru:
            for track in group_track:
                track_code = self.encode(torch.from_numpy(track))
                HS.append(track_code)
            hidden_state = torch.stack(HS,dim=1)
            return hidden_state
        else:
            CS = []
            for track in group_track:
                track_code = self.encode(torch.from_numpy(track))
                HS.append(track_code[0])
                CS.append(track_code[1])
            hidden_state = torch.stack(HS,dim=1)
            cell_state = torch.stack(CS,dim=1)

            return hidden_state,cell_state

class SGAN_PoolingNet(nn.Module):
    '''
    single core
    input: 
        group_track: encoded track
        hidden_state: size=[1,batch,hidden_size]
    output:
        hidden_state_after_group_pooling: for decoder to generate a sequence, size=[1,batch,hidden_size]
    '''
    def __init__(self, encoder, trial=0) -> None:
        super(SGAN_PoolingNet,self).__init__()
        self.hidden_size = encoder.hidden_size
        self.embdding_layer = encoder.embadding

        dim_list = [encoder.embadding_size+self.hidden_size,2*self.hidden_size,self.hidden_size]
        self.mlp = make_mlp(dim_list)
    
    def pooling_one(self,hidden_state,group_track,center_idx_local):
        center_pos_end = group_track[center_idx_local][-1]
        rel_nei_end_pos = []
        for track in group_track:
            rel_end_pos = torch.from_numpy(track[-1] - center_pos_end)
            rel_nei_end_pos.append(rel_end_pos)
        REP = torch.stack(rel_nei_end_pos,dim=0)
        
        real_pos_track_embdding = self.embdding_layer(REP)
        real_pos_track_embdding = torch.unsqueeze(real_pos_track_embdding,dim=0)
        mlp_input = torch.concatenate((hidden_state,real_pos_track_embdding),dim=2)[0]
        out = self.mlp(mlp_input)
        out = out.max(0)[0]
        return out,REP
    
    def forward(self,hidden_state,group_track):
        group_pooling_hidden = []
        for center_idx_local in range(len(group_track)):
            pooling_hidden,_ = self.pooling_one(hidden_state,group_track,center_idx_local)
            group_pooling_hidden.append(pooling_hidden)
        GPH = torch.stack(group_pooling_hidden,dim=0)

        return torch.unsqueeze(GPH,0)

class SLSTM_SPooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,hidden_state,group_track,state_tuple):
        if self.ifgru:
            _,state_tuple = self.rnn(hidden_state)
            decoder_h = state_tuple
        else:
            _,state_tuple = self.rnn(hidden_state)
            decoder_h = state_tuple[0]
        
        return 1

class SGAN_decoder(nn.Module):
    '''
    input:
        center_hidden_state:
        pooling_tyep: SocialPooling(Social-LSTM), PoolingNet(Social-GAN)
        pidx_list:
        center_pidx_local:
    output:
        predic_track:
    '''
    def __init__(self, encoder, SPoolingNet, trial=0) -> None:
        super(SGAN_decoder,self).__init__()
        hidden_size = encoder.hidden_size
        embadding_size = encoder.embadding_size
        self.ifgru = encoder.ifgru
        self.spooling = SPoolingNet
        self.predict_length = 12
        self.embadding = encoder.embadding
        self.pooling_p_step = 1

        if self.ifgru:
            self.decode = nn.GRU(input_size = embadding_size,hidden_size = hidden_size,num_layers = 1)
        else:
            self.decode = nn.LSTM(input_size = embadding_size,hidden_size = hidden_size,num_layers = 1)
        pass

        self.hidden2pos = nn.Linear(in_features=hidden_size,out_features=2)

    def forward(self,hidden_state,group_track):
        group_pooling_hidden = self.spooling(hidden_state,group_track)
        pos = self.hidden2pos(group_pooling_hidden)

        GTN = [] # group track new
        for idx,track in enumerate(group_track):
            GTN.append(np.r_[track,np.array(torch.detach(pos[0][idx].unsqueeze(0)))])
        group_track = GTN

        pos_embadding = self.embadding(pos)
        output,state_tuple = self.decode(pos_embadding)

        for _ in range(self.predict_length):
            # socail pooling
            if self.pooling_p_step:
                group_pooling_hidden = self.spooling(output,group_track)
            else:
                group_pooling_hidden = state_tuple

            pos = self.hidden2pos(group_pooling_hidden)

            # update group track
            GTN = [] # group track new
            for idx,track in enumerate(group_track):
                GTN.append(np.r_[track,np.array(torch.detach(pos[0][idx].unsqueeze(0)))])
            group_track = GTN

            # position embadding for prediction
            pos_embadding = self.embadding(pos)
            
            # rnn operation
            output,state_tuple = self.decode(pos_embadding,state_tuple)

        return group_track

class SGAN_generator(nn.Module):
    def __init__(self,Encoder,SPoolingNet,Decoder,trial=0) -> None:
        super(SGAN_generator,self).__init__()
        self.encoder = Encoder
        self.SPoolingNet = SPoolingNet
        # self.decoder = nn.Linear(in_features=self.encoder.hidden_size,out_features=2)
        self.decoder = Decoder

    def forward(self,meta_item,set_file):
        # gain the track in the scene of last frame
        group_track = search_group_track_pos(meta_item,set_file)

        # encoder the track
        if self.encoder.ifgru:
            hidden_state = self.encoder(group_track)
        else:
            hidden_state,cell_state = self.encoder(group_track)

        # decode
        prediction_track = self.decoder(hidden_state,group_track)

        return prediction_track

class SGAN_discriminator(nn.Module):
    def __init__(self, trial=0) -> None:
        super().__init__()
    def forward(self):
        return 1
    
class SGAN_dataset(Dataset):
    def __init__(self,item_idx) -> None:
        super().__init__()
        self.item = item_idx
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
    EPOCHS = trial.suggest_int("EPOCHS",1,10)

    # [train_set,validation_set,test_set]
    train_valid_array = np.load('./data/meta/train_valid.npy')
    data_div_para = trial.suggest_categorical("Cross Validation k", [5,10,20])

    train_validation_idx = data_divide(train_valid_array,para=data_div_para)

    Encoder = SGAN_encoder(trial)
    SocialPooling = SGAN_PoolingNet(trial)
    Decoder = SGAN_decoder(trial)

    optimizer = getattr(torch.optim, optimizer_name)(Encoder.parameters(), lr=lr)
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
                Encoder = train(net=Encoder,train_loader=train_loader,criterion=criterion,
                      optimizer=optimizer,batch_size=batch_size)
                # validation
                epoch_error,_ = valid(Encoder,valid_loader,criterion)

                valid_error = torch.concat((valid_error,epoch_error))
            valid_error = torch.tensor([valid_error.mean()])

            # drop trail if error is not acceptable
            trial.report(valid_error.item(),CV_i)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            else:
                optuna_error = torch.concat((valid_error,optuna_error))

        optuna_error = optuna_error.mean()

    torch.save(Encoder.state_dict(),'./model/LinearDMS/trial/trial_{}.model'.format(trial.number))
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