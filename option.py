import os
import pickle
import numpy as np

import optuna
import torch
import torch.nn as nn

from clstp.utils import data_preprocess
from model.LinearDMS.LinearDMS_utils import LinearDMS_test, LinearDMS_data_loader, LinearDMS_net, LinearDMS_obj

def main():
    if os.path.exists(''.join(['./data/meta/','meta_info.pkl'])) is False:
        initialization()

    trial = optuna_study()
    dic_path = ''.join(['./model/LinearDMS/trial_',trial.number,'.model'])
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
    study = optuna.create_study(direction='minimize',study_name='LinearDMS')
    study.optimize(LinearDMS_obj,n_trials=trial_num)
    with open('./model/LinearDMS/study_linear.pkl','wb') as path:
        pickle.dump(study,path)
        path.close()
    
    return study.best_trial

def Model_test(trial,net_dic,test_data_path='./data/meta/test_array.npy'):
    net = LinearDMS_net(trial)
    net_state = torch.load(net_dic)
    net.load_state_dict(net_state)

    criterion = nn.MSELoss()
    test_item_idx = np.load(test_data_path)
    test_loader = LinearDMS_data_loader(test_item_idx=test_item_idx)
    error_table = LinearDMS_test(net,test_loader,criterion)
    
    return error_table

if __name__=='__main__':
    # main()
    pass