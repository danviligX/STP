import numpy as np
import optuna
import torch
import torch.nn as nn

from clstp.dataio import read_set_file, stp_dataloader
from clstp.utils import Args, data_divide, search_group_track_pos

class mpp_net(nn.Module):
    def __init__(self,args) -> None:
        super(mpp_net,self).__init__()
        self.args = args
        self.embadding_size = args.embadding_size
        self.pre_length = args.pre_length
        self.hidden_size = args.hidden_size
        self.max_nei_dis = 5
        self.input_len = 8
        self.rnn_type = 1

        self.embadding = nn.Linear(in_features=2,out_features=self.embadding_size)

        if self.rnn_type:
            self.cell = nn.LSTMCell(input_size=self.hidden_size+self.embadding_size,hidden_size=self.hidden_size)
        else:
            self.cell = nn.GRUCell(input_size=self.hidden_size+self.embadding_size,hidden_size=self.hidden_size)
        
        self.deembadding = nn.Linear(in_features=self.hidden_size,out_features=2)
    
    def forward(self,group_track):
        cell_state = self.group_encode(group_track)
        if self.rnn_type:
            hidden = torch.clone(cell_state[0])
        else:
            hidden = torch.clone(cell_state)
        
        for _ in range(self.pre_length):
            Cel_Input = []
            for center_idx in range(len(group_track)):
                S_hidden = self.decode_social_pooling(hidden=hidden,group_track=group_track,center_idx=center_idx)
                pos_emabadding = self.embadding(group_track[center_idx][-1]).unsqueeze(0)

                cell_input = torch.concat((pos_emabadding,S_hidden),dim=1)
                Cel_Input.append(cell_input)
            
            Cel_Input = torch.concat(Cel_Input,dim=0)
            cell_state = self.cell(Cel_Input,cell_state)

            if self.rnn_type:
                hidden = torch.clone(cell_state[0])
            else:
                hidden = torch.clone(cell_state)

            pos = self.deembadding(hidden)

            T = []
            for idx,track in enumerate(group_track):
                track = torch.concat((track,track[-1]+pos[idx].unsqueeze(0)),dim=0)
                T.append(track)
            group_track = T

        return group_track[0]
    
    def group_encode(self,group_track):
        init_pos = []
        for track in group_track:
            init_pos.append(track[0])
        init_pos = torch.stack(init_pos,dim=0)
        

        pos_embadding = self.embadding(init_pos)
        zero_init_ten = torch.zeros([len(group_track),self.hidden_size]).to(self.args.device)
        cell_state = self.cell(torch.concat((pos_embadding,zero_init_ten),dim=1))

        # [num,hidden_size], used for iteration
        if self.rnn_type:
            # IS = cell_state[0]
            S_hidden = cell_state[0]
        else:
            # IS = cell_state
            S_hidden = cell_state

        for frame_idx in range(self.input_len-1):
            # pick neighbors out 
            NEI = []
            for nei_idx in range(len(group_track)):
                if len(group_track[nei_idx]) >= self.input_len - frame_idx: NEI.append(nei_idx)
            
            # update the neighors' social hidden state
            NEI_pos = []
            NEI_sh = []
            for nei_idx in NEI:
                nei_frame_idx = len(group_track[nei_idx]) - self.input_len + frame_idx
                nei_pos = group_track[nei_idx][nei_frame_idx]
                single_social_h = self.encode_social_pooling(center_pos=nei_pos,hidden=S_hidden,group_track=group_track,
                                                  NEI=NEI,frame_idx=frame_idx)
                NEI_sh.append(single_social_h)
                NEI_pos.append(nei_pos)

            NEI_pos = torch.stack(NEI_pos,dim=0)
            used_h = torch.concat(NEI_sh,dim=0)
            pos_embadding = self.embadding(NEI_pos)
            cell_input = torch.concat((pos_embadding,used_h),dim=1)

            # mask for cell_state
            if self.rnn_type:
                masked_cell_state = (cell_state[0][NEI],cell_state[1][NEI])
                masked_cell_state = self.cell(cell_input,masked_cell_state)

                h_state = self.cell_state_concat(masked_cell_state[0],cell_state[0],NEI)
                c_state = self.cell_state_concat(masked_cell_state[1],cell_state[1],NEI)

                cell_state = (h_state,c_state)
                S_hidden = cell_state[0]
            else:
                masked_cell_state = cell_state[NEI]
                masked_cell_state = self.cell(cell_input,masked_cell_state)
                h_state = self.cell_state_concat(masked_cell_state,cell_state,NEI)
                cell_state = h_state
                S_hidden = cell_state
            
        return cell_state
    
    def cell_state_concat(self,masked_cell_state,cell_state,NEI):
        '''
        take GRU cell_state as example
        '''
        CS = []
        for idx in range(len(cell_state)):
            if idx not in NEI: 
                CS.append(cell_state[idx])
            else:
                idx_local = torch.argwhere(torch.tensor(NEI) == idx)
                CS.append(masked_cell_state[idx_local].squeeze())

        CS = torch.stack(CS,dim=0)
        return CS

    def encode_social_pooling(self,center_pos,hidden,group_track,NEI,frame_idx):
        '''
        Pooling socail hidden_state into the center's hidden
        input:
            frame_idx: the center_pos frame_idx, to calculate the neighbors' frame_idx
        output:
            SH.sum(dim=0): mean pooling hidden_state, [1,hidden_size]
        '''
        social_hidden = []
        for nei_idx in NEI:
            nei_frame_idx = len(group_track[nei_idx]) - self.input_len + frame_idx
            rel_pos = group_track[nei_idx][nei_frame_idx] - center_pos
            rel_dis = torch.norm(rel_pos).item()
            if rel_dis < self.max_nei_dis: social_hidden.append(hidden[nei_idx])
        SH = torch.stack(social_hidden,dim=0)
        return SH.sum(dim=0).unsqueeze(0)
    
    def decode_social_pooling(self,hidden,group_track,center_idx):
        center_pos = group_track[center_idx][-1]
        SH = []
        for idx,track in enumerate(group_track):
            rel_pos = track[-1] - center_pos
            rel_dis = torch.norm(rel_pos).item()
            if rel_dis < self.max_nei_dis: SH.append(hidden[idx])
        S_hidden = torch.stack(SH,dim=0)

        return S_hidden.sum(dim=0).unsqueeze(0)
    
def mpp_obj(trial):
    args = Args()

    # cuda
    if torch.cuda.is_available():
        args.device = torch.device("cuda:0")
    else:
        args.device = torch.device("cpu")

    # net initialization parameters
    args.model_name = 'MPP'
    args.embadding_size = trial.suggest_int("embadding_size", 8, 128,step=8)
    args.hidden_size = trial.suggest_int("hidden_size", 32, 512,step=16)
    args.pre_length = 12

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

    net = mpp_net(args=args).to(device=args.device)
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
            group_track_input = search_group_track_pos(meta_item,set_file,fram_num=8,device=args.device)
            group_track_target = search_group_track_pos(meta_item,set_file,fram_num=20,device=args.device)

            # forward
            out = net(group_track_input)
            loss = criterion(group_track_target[0][8:],out[8:])

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