import os
import pickle
import sys

import numpy as np

from clstp.dataio import (boot_strapping, cross_validation, generate_meta_info,
                          generate_set_info, hould_out)

# raw_file_name = os.path.splitext(raw_file_name)[0]
# raw_file_extension = os.path.splitext(raw_file)[-1]

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
    step 3: if the item number of left data is not divisible by 20, select more data into test set.
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
        test_set = np.union1d(test_set,select_set_i_index).astype(np.uint32)
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
    meta_info = np.zeros([1,4]).astype(np.uint32)
    fid_unique_list = []
    for i in range(len(raw_file_name_list)):
        raw_file_name = raw_file_name_list[i]
        raw_file_path = os.path.join(raw_folder_path,raw_file_name)

        meta_item,fid_unique = generate_meta_info(raw_file_path,raw_file_name_list)
        meta_info = np.concatenate((meta_info,meta_item))
        fid_unique_list.append(fid_unique)
    
    meta_info = np.delete(meta_info,0,axis=0)

    return meta_info,fid_unique_list

def search_track_pos(meta_item,set_file):
    '''
    input:
        meta_item: [set_code,pidx,start_fidx,end_fidx] a item of meta_info
        set_file: frame_info.pkl
    output: shape(2,end_fidx-start_fidx)
    '''
    pidx = meta_item[1]
    start_fidx = meta_item[2]
    end_fidx = meta_item[3]
    frame_range = range(start_fidx,end_fidx)
    track = np.zeros([len(frame_range),2]).astype(np.float32)
    for idx,frame_idx in enumerate(frame_range):
        frame_info = set_file[frame_idx]
        local_idx = np.argwhere(frame_info[:,0]==pidx).item()
        pos = frame_info[local_idx,1:]
        track[idx,:] = pos

    return track