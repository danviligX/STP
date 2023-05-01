import json
import os
import sys

import numpy as np
import pickle


def generate_set_info(raw_file_path):
    '''
    input: raw_file
    output: set_file: [ped_idx, pos_x, pos_y] => set_file_path
    '''
    pid_list,fid_list,pos_list = track_list(raw_file_path)
    pid_unique = np.unique(pid_list)
    Fun_len_fid_list = len(fid_list)
    i = 0
    frame_list = []
    while i < Fun_len_fid_list-1:
        fid = fid_list[i]
        pidx_pos_list = []
        for j in range(Fun_len_fid_list-i):
                temp_fid = fid_list[i+j]
                if temp_fid != fid: break
                pid = pid_list[i+j]
                pos = pos_list[i+j]
                pidx_pos = [np.argwhere(pid_unique==pid)[0,0],pos[0],pos[1]]
                pidx_pos_list.append(pidx_pos)
        
        pidx_pos_list = np.array(pidx_pos_list).astype(np.float32)
        frame_list.append(pidx_pos_list)
        i += j 

    return frame_list,pid_unique

def generate_meta_info(raw_file_path, raw_file_name_list=None):
    '''
    input: raw_file, raw_file_mapping_table
    output: meta_info: [file_code, pedestrain_idx, start_frame_idx, end_frame_idx]
    '''
    track_pid_list,fid_list,_ = track_list(raw_file_path)
    pid_list,fse_list,_ = scene_list(raw_file_path)

    raw_file_name = os.path.split(raw_file_path)[1]
    if raw_file_name_list is None:
        data_code = 0
    else:
        data_code = raw_file_name_list.index(raw_file_name)

    pid_unique = np.unique(track_pid_list)
    fid_unique = np.unique(fid_list)

    meta_info = np.zeros([len(pid_list),4])
    for i in range(len(pid_list)):
        pid_idx = np.argwhere(pid_unique==pid_list[i])[0,0]
        start_fid_idx = np.argwhere(fid_unique==fse_list[i][0])[0,0]
        end_fid_idx = np.argwhere(fid_unique==fse_list[i][1])[0,0]

        temp_vec = [data_code,pid_idx,start_fid_idx,end_fid_idx]
        meta_info[i,:] = temp_vec

    meta_info = meta_info.astype(np.uint32)
   
    return meta_info,fid_unique

def track_list(raw_file_path):
    file = open(raw_file_path,'r')

    fid_list = []
    pid_list = []
    pos_list = []

    json_loads  = json.loads
    for line in file:
        line = json_loads(line)
        track = line.get('track')
        
        if track is not None:
            fid_list.append(track['f'])
            pid_list.append(track['p'])
            pos_list.append([track['x'],track['y']])

    # sort the data by frame id
    sorted_id = sorted(range(len(fid_list)),key=lambda k:fid_list[k], reverse=False)
    pos_list = np.array(pos_list)[sorted_id,:]
    pid_list = np.array(pid_list)[sorted_id]
    fid_list = np.array(fid_list)[sorted_id]

    return pid_list,fid_list,pos_list

def scene_list(raw_file_path):
    file = open(raw_file_path,'r')

    pid_list = []
    fse_list = []
    tag_list = []

    json_loads  = json.loads
    for line in file:
        line = json_loads(line)
        scene = line.get('scene')

        if scene is not None:
            pid_list.append(scene['p'])
            fse_list.append([scene['s'],scene['e']])
            tag_list.append(scene['tag'])
    
    return pid_list,fse_list,tag_list

def cross_validation(item_index_array,k=10):
    '''
    input: item_index_array
    output: list of set item index
    '''
    np.random.shuffle(item_index_array)
    set_list = item_index_array.reshape((k,-1))
    train_valid_set = []
    for i in range(k):
        train_set = np.delete(set_list,i,axis=0).reshape((1,-1))[0].astype(np.uint32)
        validation_set = set_list[i].reshape((1,-1))[0].astype(np.uint32)
        train_valid_set.append((train_set,validation_set))
    return train_valid_set

def boot_strapping(item_index_array,num=[2000,20]):
    train_num = num[0]
    validation_num=num[1]
    train_set = np.random.choice(item_index_array,size=train_num,replace=True).astype(np.uint32)
    valid_set = np.random.choice(item_index_array,size=validation_num,replace=True).astype(np.uint32)
    return [train_set,valid_set]

def hould_out(item_index_array,rate = 0.1):
    if rate<0 or rate > 1:
        print('Hould out rate error!')
        sys.exit()
    np.random.shuffle(item_index_array)
    valid_split_num = int(len(item_index_array)*rate)
    valid_set = item_index_array[:valid_split_num].astype(np.uint32)
    train_set = item_index_array[valid_split_num+1:].astype(np.uint32)
    return [train_set,valid_set]

def read_set_file(set_folder_path = './data/set/',
                  set_name_list_path = './data/meta/data_name_list.pkl',
                  ):

    with open(set_name_list_path,'rb') as path:
        data_name_list = pickle.load(path)
        path.close()
    
    set_file_list = []
    for i in range(len(data_name_list)):
        data_name = data_name_list[i]
        with open("".join([set_folder_path,data_name,'.pkl']),'rb') as path:
            file = pickle.load(path)
            path.close()
        set_file_list.append(file)

    return set_file_list