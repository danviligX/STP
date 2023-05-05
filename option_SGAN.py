import os
import pickle

import numpy as np
import optuna
import torch
import torch.nn as nn

from clstp.dataio import read_set_file
from clstp.utils import data_preprocess
from model.SGAN.sgan_utils import *


def main():
    if os.path.exists(''.join(['./data/meta/','meta_info.npy'])) is False:
        initialization()

    trial = optuna_study()
    dic_path = ''.join(['./model/SGAN/trial_',trial.number,'.model'])
    error_table = Model_test(trial,dic_path)
    print(error_table)

def initialization():
    # preprocess data, save meta and set file into SSD
    data_preprocess(
        raw_folder_path='./data/raw/',
        set_folder_path='./data/set/',
        meta_folder_path='./data/meta/',
        test_set_rate=0.1)

def optuna_study(trial_num=5):
    study = optuna.create_study(direction='minimize',study_name='SGAN')
    study.optimize(SGAN_obj,n_trials=trial_num)
    with open('./model/SGAN/study.pkl','wb') as path:
        pickle.dump(study,path)
        path.close()
    
    return study.best_trial

def Model_test(trial,net_dic,test_data_path='./data/meta/test_array.npy'):
    net = SGAN_generator(trial)
    net_state = torch.load(net_dic)
    net.load_state_dict(net_state)

    criterion = nn.MSELoss()
    test_item_idx = np.load(test_data_path)
    test_loader = SGAN_data_loader(test_item_idx=test_item_idx)
    set_file_list = read_set_file()
    error_table = SGAN_test(net,test_loader,criterion,set_file_list)
    
    return error_table

if __name__=='__main__':
    main()
    pass