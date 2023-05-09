import os
import pickle
import sys

import numpy as np
import torch.nn as nn
import torch

from clstp.dataio import (boot_strapping, cross_validation, generate_meta_info,
                          generate_set_info, hould_out)

# raw_file_name = os.path.splitext(raw_file_name)[0]
# raw_file_extension = os.path.splitext(raw_file)[-1]
class Args(object):
    def __init__(self) -> None:
        pass

class stp_poolingnet(nn.Module):
    '''
    single core
    input: 
        group_track: shape = [length,2]
        hidden_state: size=[1,batch,hidden_size]
    output:
        socail_hidden: for decoder to generate a sequence, size=[batch,hidden_size]
    '''
    def __init__(self, args) -> None:
        super(stp_poolingnet,self).__init__()
        self.hidden_size = args.hidden_size
        self.embdding_size = args.embadding_size
        self.mlp_hidden_size = args.PNMLP_hidden_size
        
        self.embadding = nn.Linear(in_features=2,out_features=self.embdding_size)

        dim_list = [self.embdding_size+self.hidden_size,self.mlp_hidden_size,self.hidden_size]
        self.mlp = make_mlp(dim_list,batch_norm=False)
    
    def pooling_one(self,hidden_state,group_track,center_idx_local):
        # calculate relative end position
        center_pos_end = group_track[center_idx_local][-1]
        rel_nei_end_pos = []
        for track in group_track:
            rel_end_pos = track[-1] - center_pos_end
            rel_nei_end_pos.append(rel_end_pos)
        REP = torch.stack(rel_nei_end_pos,dim=0)
        
        real_pos_track_embdding = self.embadding(REP)
        mlp_input = torch.concatenate((hidden_state,real_pos_track_embdding),dim=1)
        out = self.mlp(mlp_input)
        out = out.max(0)[0]
        return out
    
    def forward(self,hidden_state,group_track):
        group_pooling_hidden = []
        for center_idx_local in range(len(group_track)):
            pooling_hidden = self.pooling_one(hidden_state,group_track,center_idx_local)
            group_pooling_hidden.append(pooling_hidden)
        GPH = torch.stack(group_pooling_hidden,dim=0)

        return GPH

class stp_attention_pooling(nn.Module):
    '''
    single core
    input: 
        group_track: shape = [length,2]
        hidden_state: size=[1,batch,hidden_size]
    output:
        socail_hidden: for decoder to generate a sequence, size=[batch,hidden_size]
    '''
    def __init__(self, args) -> None:
        super(stp_attention_pooling,self).__init__()
        self.hidden_size = args.hidden_size
        self.embdding_size = args.embadding_size
        self.mlp_hidden_size = args.PNMLP_hidden_size
        self.rel_mlp_hidden_size = args.rel_mlp_hidden_size
        self.abs_mlp_hidden_size = args.abs_mlp_hidden_size
        self.attention_mlp_hidden_size = args.attention_mlp_hidden_size
        
        
        self.embadding = nn.Linear(in_features=2,out_features=self.embdding_size)

        dim_list = [self.embdding_size+self.hidden_size,self.mlp_hidden_size,self.hidden_size]
        self.mlp = make_mlp(dim_list)

        # attention mlp
        rel_dim_list = [2,self.rel_mlp_hidden_size,1]
        abs_dim_list = [2,self.abs_mlp_hidden_size,1]
        attention_dim_list = [4,self.attention_mlp_hidden_size,1]

        self.rel_mlp = make_mlp(rel_dim_list,batch_norm=False)
        self.abs_mlp = make_mlp(abs_dim_list,batch_norm=False)
        self.attention_mlp = make_mlp(attention_dim_list,batch_norm=False)

        self.softmax = nn.Softmax(dim=1)
    
    def attention_matrix_line(self,hidden_state,group_track,center_idx_local):
        if len(hidden_state) == 1: return hidden_state

        # calculate relative end position
        center_pos_end = group_track[center_idx_local][-1]
        center_pos_bfend = group_track[center_idx_local][-2]

        rel_nei_end_pos = []
        rel_nei_bfend_pos = []
        abs_nei_end_pos = []
        abs_center_end_pos = []
        for track in group_track:
            rel_end_pos = track[-1] - center_pos_end
            rel_bfend_pos = track[-2] - center_pos_bfend
            rel_nei_end_pos.append(rel_end_pos)
            rel_nei_bfend_pos.append(rel_bfend_pos)
            abs_nei_end_pos.append(track[-1])
            abs_center_end_pos.append(center_pos_end) # clone the position

        REP = torch.stack(rel_nei_end_pos,dim=0) # relative end position
        RBEP = torch.stack(rel_nei_bfend_pos,dim=0) # relative penultimate position
        RV = REP - RBEP # relative velocity
        ANP = torch.stack(abs_nei_end_pos,dim=0) # abstract neighors position
        ACP = torch.stack(abs_center_end_pos,dim=0) # abstract center position

        rel_fea = torch.stack((REP,RV),dim=0) # [feature_num,batch,feature_size], [2,nei_num,2]
        abs_fea = torch.stack((ANP,ACP),dim=0)

        # attention weight calculation
        rel_attention = self.rel_mlp(rel_fea)
        abs_attention = self.abs_mlp(abs_fea)
        attention_fea = torch.concat((rel_attention,abs_attention),dim=0).transpose(0,2)
        metrc_dis = self.attention_mlp(attention_fea).squeeze().unsqueeze(0) # [1,nei_num]
        metrc_dis = self.softmax(metrc_dis)

        out = torch.matmul(metrc_dis,hidden_state) # [1,hidden_size]
        return out
    
    def forward(self,hidden_state,group_track):
        group_pooling_hidden = []
        for center_idx_local in range(len(group_track)):
            pooling_hidden = self.attention_matrix_line(hidden_state,group_track,center_idx_local)
            group_pooling_hidden.append(pooling_hidden)
        GPH = torch.concat(group_pooling_hidden,dim=0)

        return GPH
    
def data_preprocess(
        raw_folder_path='./data/raw/',
        set_folder_path='./data/set/',
        meta_folder_path='./data/meta/',
        test_set_rate=0.1):

    if os.path.exists(''.join([raw_folder_path,'raw_file_used.pkl'])):
        with open(''.join([raw_folder_path,'raw_file_used.pkl']),'rb') as path:
            raw_file_name_list = pickle.load(path)
            path.close()
    else:
        raw_file_name_list = os.listdir(raw_folder_path)
        with open(''.join([raw_folder_path,'raw_file_used.pkl']),'wb') as path:
            pickle.dump(raw_file_name_list,path,protocol=2)
            path.close()
    
    # generate meta info
    meta_info,fid_unique_list = meta_generator(raw_folder_path,raw_file_name_list)
    train_valid_array,test_array = meta_divide(meta_info,rate=test_set_rate)

    np.save("".join([meta_folder_path,'meta_info']),meta_info)
    np.save("".join([meta_folder_path,'train_valid']),train_valid_array)
    np.save("".join([meta_folder_path,'test_array']),test_array)

    with open("".join([meta_folder_path,'frame_map.pkl']),'wb') as path:
        pickle.dump(fid_unique_list,path,protocol=2)
        path.close()

    data_name_list = []
    for i in range(len(raw_file_name_list)):
        # get the data name(with out extension)
        raw_file_name = raw_file_name_list[i]
        data_name = os.path.splitext(raw_file_name)[0]
        data_name_list.append(data_name)

        frame_list,pid_unique = generate_set_info(os.path.join(raw_folder_path,raw_file_name))

        # save the set info and pid map, frame map,
        with open("".join([os.path.join(set_folder_path,data_name),'.pkl']),'wb') as path:
            pickle.dump(frame_list,path,protocol=2)
            path.close()
        np.save("".join([os.path.join(set_folder_path,data_name),"_pid_map"]),pid_unique)

    with open("".join([meta_folder_path,'data_name_list.pkl']),'wb') as path:
        pickle.dump(data_name_list,path,protocol=2)
        path.close()

def meta_divide(meta_info,rate=0.1):
    '''
    step 1: pick up 10% items from each dataset
    step 2: the left data is the train and valid set
    step 3: if the item number of left data is not divisible by 20, select more data into test set. For cross validation with k=5,10,20
    '''
    meta_info_array = np.arange(meta_info[:,0].size)
    _,set_label_count = np.unique(meta_info[:,0],return_counts=True)
    array_index = np.concatenate((np.array([0]),set_label_count))

    # step 1
    test_set = []
    for i in range(set_label_count.size):
        set_i_array = meta_info_array[sum(array_index[:i+1]):sum(array_index[:i+2])]
        select_set_i_index = np.random.choice(set_i_array,size=int(array_index[i+1]*rate),replace=False)
        test_set = np.union1d(test_set,select_set_i_index)

    # step 2
    train_valid_set = np.setdiff1d(meta_info_array,test_set)

    # step 3
    res = train_valid_set.size%20
    if res!= 0:
        select_set_i_index = np.random.choice(train_valid_set,size=res,replace=False)
        test_set = np.union1d(test_set,select_set_i_index).astype(np.int32)
        train_valid_set = np.setdiff1d(train_valid_set,test_set)
    return train_valid_set,test_set

def data_divide(item_index_array,method='CV',para=10):
    
    if method=='HO':
        train_valid_set = hould_out(item_index_array,rate=para)
    elif method=='CV':
        train_valid_set = cross_validation(item_index_array,k=para)
    elif method=='BS':
        train_valid_set = boot_strapping(item_index_array,num=para)
    else:
        print('Data divide method error!')
        sys.exit()
    return train_valid_set

def meta_generator(raw_folder_path,raw_file_name_list):
    meta_info = np.zeros([1,4]).astype(np.int32)
    fid_unique_list = []
    for i in range(len(raw_file_name_list)):
        raw_file_name = raw_file_name_list[i]
        raw_file_path = os.path.join(raw_folder_path,raw_file_name)

        meta_item,fid_unique = generate_meta_info(raw_file_path,raw_file_name_list)
        meta_info = np.concatenate((meta_info,meta_item))
        fid_unique_list.append(fid_unique)
    
    meta_info = np.delete(meta_info,0,axis=0)

    return meta_info,fid_unique_list

def search_track_pos(meta_item,set_file,search_pidx,device=torch.device('cpu')):
    '''
    input:
        meta_item: [set_code,pidx,start_fidx,end_fidx] a item of meta_info
        set_file: frame_info.pkl
    output: 
        track: array, shape(2,end_fidx-start_fidx)
        pidx_neighbot_array: list of neighbor pidx
    '''
    pidx = search_pidx
    start_fidx = meta_item[2]
    end_fidx = meta_item[3]
    frame_range = range(start_fidx,end_fidx)
    track = np.zeros([len(frame_range),2]).astype(np.float32)
    start_idx_local = -1
    for idx,frame_idx in enumerate(frame_range):
        frame_info = set_file[frame_idx]
        local_idx = np.argwhere(frame_info[:,0]==pidx)
        if local_idx.size == 0:
            start_idx_local = -1
            continue
        else:
            pos = frame_info[local_idx,1:]
            track[idx,:] = pos
            if start_idx_local == -1:
                start_idx_local = idx

    pidx_neighbor_array = np.delete(frame_info[:,0],local_idx,axis=0)
    track = np.delete(track,np.arange(start_idx_local),axis=0)
    
    T = torch.from_numpy(track)
    if device==torch.device('cpu'):
        return T.cpu(),pidx_neighbor_array
    else:
        return T.cuda(),pidx_neighbor_array
    
def search_group_track_pos(meta_item,set_file,device=torch.device('cpu'),fram_num=20):
    '''
    input:
        meta_item: [set_code,pidx,start_fidx,end_fidx] a item of meta_info
        set_file: frame_info.pkl
    output:
        group_track: a list of track in the last frame of the scene, the first one is the center pidx's track
    '''
    meta_item = np.array(meta_item)
    meta_item[3] = meta_item[2] + fram_num
    group_track,pidx_neighbor_array = search_track_pos(meta_item,set_file,meta_item[1],device=device)
    group_track = [group_track]
    for pidx in pidx_neighbor_array:
        track,_ = search_track_pos(meta_item,set_file,pidx,device=device)
        if len(track) > 1:
            group_track.append(track)
        
    meta_item[3] = meta_item[2] + 20
    return group_track

def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)