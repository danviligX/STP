import numpy as np
import optuna
import torch
import torch.nn as nn

from clstp.dataio import read_set_file, stp_dataloader
from clstp.utils import Args, data_divide, search_group_track_pos, stp_attention_pooling, make_mlp


class sa_net(nn.Module):
    def __init__(self,args) -> None:
        super(sa_net,self).__init__()
        self.embadding_size = args.embadding_size
        self.pre_length = args.pre_length
        self.hidden_size = args.hidden_size
        self.rnn_type = 0
        self.PoolingNet = stp_attention_pooling(args=args)
        self.pre_mlp_hidden_size = args.pre_mlp_hidden_size
        self.Pooling_per_step = 0

        self.embadding = nn.Linear(in_features=2,out_features=self.embadding_size)
        if self.rnn_type:
            self.encoder = nn.LSTM(input_size=self.embadding_size,hidden_size=self.hidden_size)
            self.decoder = nn.LSTM(input_size=self.embadding_size,hidden_size=self.hidden_size)
        else:
            self.encoder = nn.GRU(input_size=self.embadding_size,hidden_size=self.hidden_size)
            self.decoder = nn.GRU(input_size=self.embadding_size,hidden_size=self.hidden_size)
        
        self.deembadding = nn.Linear(in_features=self.hidden_size,out_features=2)
        
        dim_list = [self.hidden_size,self.pre_mlp_hidden_size,2]
        self.pre_mlp = make_mlp(dim_list,batch_norm=False)

    def forward(self,group_track):

        group_hidden,group_state = self.group_encode(group_track)
        group_state = self.social_poolingNet(group_hidden,group_state,group_track) # [batch,hidden_size]

        if self.Pooling_per_step:
            for _ in range(self.pre_length):
                group_state,group_track = self.group_decode(group_state,group_track)
                group_state = self.social_poolingNet(group_hidden,group_state,group_track)
        else:
            for _ in range(self.pre_length):
                group_state,group_track = self.group_decode(group_state,group_track)

        return group_track[0]
    
    def social_poolingNet(self,group_hidden,group_state,group_track):
        group_hidden = self.PoolingNet(group_hidden,group_track)
        GS = []
        if self.rnn_type:
            for idx in range(len(group_state)):
                GS.append((group_hidden[idx],group_state[idx][1]))
        else:
            for idx in range(len(group_state)):
                GS.append(group_hidden[idx])
        return GS

    def group_encode(self,group_track):
        group_ST = []
        HS = []
        for track in group_track:
            track_code,state_tuple = self.encoder(self.embadding(track))
            hidden = track_code[-1]
            group_ST.append(state_tuple)
            HS.append(hidden)
        hidden_state = torch.stack(HS,dim=0)
        return hidden_state,group_ST
    
    def group_decode(self,group_state,group_track):
        GS = []
        GT = []
        if self.rnn_type:
            for idx in range(len(group_state)):
                hidden,state_tuple = self.decoder(self.embadding(group_track[idx][-1].unsqueeze(0)),(group_state[idx][0],group_state[idx][1]))
                pos = self.pre_mlp(hidden)
                GT.append(torch.concat((group_track[idx],pos),dim=0))
                GS.append((group_state[idx][0].squeeze(),group_state[idx][1]))
        else:
            for idx in range(len(group_state)):
                hidden,state_tuple = self.decoder(self.embadding(group_track[idx][-1].unsqueeze(0)),group_state[idx].unsqueeze(0))
                pos = self.pre_mlp(hidden)
                GT.append(torch.concat((group_track[idx],pos),dim=0))
                GS.append(state_tuple.squeeze())

        return GS,GT

    
def sa_obj(trial):
    args = Args()

    # cuda
    if torch.cuda.is_available():
        args.device = torch.device("cuda:0")
    else:
        args.device = torch.device("cpu")

    # net initialization parameters
    args.model_name = 'SA'
    args.embadding_size = trial.suggest_int("embadding_size", 8, 128,step=8)
    args.hidden_size = trial.suggest_int("hidden_size", 32, 512,step=16)
    args.pre_mlp_hidden_size = trial.suggest_int("pre_mlp_hidden_size", 32, 1024,step=32)
    args.pre_length = 12

    args.rel_mlp_hidden_size = trial.suggest_int("rel_mlp_hidden_size", 8, 128,step=8)
    args.abs_mlp_hidden_size = trial.suggest_int("abs_mlp_hidden_size", 8, 128,step=8)
    args.attention_mlp_hidden_size = trial.suggest_int("attention_mlp_hidden_size", 8, 128,step=8)

    # hyperparameters selection with optuna
    args.opt = trial.suggest_categorical("optimizer", ["RMSprop", "SGD", "Adam"])
    args.lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    args.batch_size = trial.suggest_int("batch_size", 4, 32,step=4)
    # args.epoch_num = trial.suggest_int("epoch_num",5,200)
    args.epoch_num = 3

    # data prepare
    train_valid_array = np.load('./data/meta/train_valid.npy')
    train_validation_idx = data_divide(train_valid_array,para=10)
    set_file_list = read_set_file()

    net = sa_net(args=args).to(device=args.device)
    opt = getattr(torch.optim, args.opt)(net.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # initialization for optuna error
    valid_error = torch.tensor([])

    # select the first part of cross validation
    train_item_idx = train_validation_idx[1][0]
    valid_item_idx = train_validation_idx[1][1]
    train_loader,valid_loader = stp_dataloader(train_item_idx=train_item_idx,
                                                              valid_item_idx=valid_item_idx,
                                                              batch_size=args.batch_size)
    
    ESS = 0
    for epoch in range(args.epoch_num):
        net = train(net=net,train_loader=train_loader,criterion=criterion,
                    optimizer=opt,args=args,set_file_list=set_file_list)

        epoch_error,_ = valid(net,valid_loader,criterion,set_file_list,device=args.device)
        
        if epoch%5==0:
            print('trial:{}, epoch:{}, loss:{}'.format(trial.number,epoch,epoch_error.item()))
            if ESS == epoch_error.item(): raise optuna.exceptions.TrialPruned()
            ESS = epoch_error.item()

        valid_error = torch.concat((valid_error,epoch_error))
        trial.report(epoch_error.item(),epoch)
        if epoch_error > 1000: raise optuna.exceptions.TrialPruned()
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        if torch.isnan(epoch_error).any(): raise optuna.exceptions.TrialPruned()
        if torch.isinf(epoch_error).any(): raise optuna.exceptions.TrialPruned()

    optuna_error = valid_error.mean()

    torch.save(net.state_dict(),''.join(['./model/',args.model_name,'/trial/trial_',str(trial.number),'.model']))
    torch.save(args,''.join(['./model/',args.model_name,'/trial/args_',str(trial.number),'.miarg']))
    return optuna_error


def train(net,train_loader,criterion,optimizer,args,set_file_list):
    net.train()
    for _,batched_meta in enumerate(train_loader):
        optimizer.zero_grad()
        for idx,meta_item in enumerate(batched_meta):
            set_file = set_file_list[meta_item[0].item()]
            group_track_input = search_group_track_pos(meta_item,set_file,fram_num=8,device=args.device)
            group_track_target = search_group_track_pos(meta_item,set_file,fram_num=20,device=args.device)

            # forward
            out = net(group_track_input)
            loss = criterion(group_track_target[0][8:],out[8:])
            # print(loss.item())

            loss.backward()
        optimizer.step()
    return net

def valid(net,valid_loader,criterion,set_file_list,device):
    error = torch.tensor([])
    net.eval()
    with torch.no_grad():
        for _,batched_one_meta in enumerate(valid_loader):
            meta_item = batched_one_meta[0]
            set_file = set_file_list[meta_item[0].item()]
            group_track_input = search_group_track_pos(meta_item,set_file,fram_num=8,device=device)
            group_track_target = search_group_track_pos(meta_item,set_file,fram_num=20,device=device)
            
            # forward
            out = net(group_track_input)
            loss = criterion(group_track_target[0][-1],out[-1])

            loss_tensor = torch.tensor([loss.item()])
            error = torch.concat((loss_tensor,error))
        epoch_error = error.mean()
        epoch_error_std = error.std()
    
    return torch.tensor([epoch_error]),epoch_error_std

def test(net,test_loader,criterion,set_file_list,device=torch.device('cpu')):
    test_length = len(test_loader)
    error = torch.zeros(test_length,2)
    with torch.no_grad():
        for item_idx,batched_one_meta in enumerate(test_loader):
            meta_item = batched_one_meta[0]
            set_file = set_file_list[meta_item[0].item()]
            group_track_input = search_group_track_pos(meta_item,set_file,fram_num=8,device=device)
            group_track_target = search_group_track_pos(meta_item,set_file,fram_num=20,device=device)

            # forward
            out = net(group_track_input)
            loss = criterion(group_track_target[0][-1],out[-1])

            error[item_idx,0] = meta_item[0].item()
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