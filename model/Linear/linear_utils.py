import numpy as np
import optuna
import torch
import torch.nn as nn

from clstp.dataio import read_set_file, stp_dataloader
from clstp.utils import Args, data_divide, search_group_track_pos,make_mlp


class linear_net(nn.Module):
    def __init__(self,args) -> None:
        super(linear_net,self).__init__()
        self.embadding_size = args.embadding_size
        self.pre_length = args.pre_length
        self.hidden_size = args.hidden_size

        self.embadding = nn.Linear(in_features=2,out_features=self.embadding_size)

        dim_list = [8,self.hidden_size,self.pre_length]
        self.mlp = make_mlp(dim_list,batch_norm=True)

        self.linear = nn.Linear(in_features=8,out_features=12)
        self.deembadding = nn.Linear(in_features=self.embadding_size,out_features=2)
    
    def forward(self,his_track):
        # seq = self.embadding(his_track)
        seq = his_track
        seq = seq.transpose(0,1)
        seq = self.linear(seq)
        out = seq.transpose(0,1)
        # out = self.deembadding(out)
        return out

def linear_obj(trial):
    args = Args()

    # cuda
    if torch.cuda.is_available():
        args.device = torch.device("cuda:0")
    else:
        args.device = torch.device("cpu")

    # net initialization parameters
    args.model_name = 'Linear'
    args.embadding_size = trial.suggest_int("embadding_size", 8, 128,step=8)
    args.hidden_size = trial.suggest_int("hidden_size", 32, 512,step=16)
    args.pre_length = 12

    # hyperparameters selection with optuna
    args.opt = trial.suggest_categorical("optimizer", ["RMSprop", "SGD", "Adam"])
    args.lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    args.batch_size = trial.suggest_int("batch_size", 4, 32,step=4)
    args.epoch_num = trial.suggest_int("epoch_num",5,50)
    # args.epoch_num = 10

    # data prepare
    train_valid_array = np.load('./data/meta/train_valid.npy')
    train_validation_idx = data_divide(train_valid_array,para=10)
    set_file_list = read_set_file()

    net = linear_net(args=args).to(device=args.device)
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
        print('trial:{}, epoch:{}, loss:{}'.format(trial.number,epoch,epoch_error.item()))
        
        if epoch%5==0:
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
            group_track_target = search_group_track_pos(meta_item,set_file,fram_num=20,device=args.device)

            # forward
            out = net(group_track_target[0][:8])
            loss = criterion(group_track_target[0][8:],out)

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
            group_track_target = search_group_track_pos(meta_item,set_file,fram_num=20,device=device)
            
            # forward
            out = net(group_track_target[0][:8])
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
            group_track_target = search_group_track_pos(meta_item,set_file,fram_num=20,device=device)

            # forward
            out = net(group_track_target[0][:8])
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