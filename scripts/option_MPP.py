import os
import sys
sys.path.append(os.getcwd())
import optuna
import torch
import numpy as np
import pickle
import torch.nn as nn
from model.MPP.mpp_utils import mpp_net,mpp_obj,test
from clstp.dataio import stp_dataloader,read_set_file

def main():
    model_name = 'MPP'
    save_path = ''.join(['./model/',model_name,'/',model_name,'_study.pkl'])

    trial = optuna_study(save_dic=save_path,study_name=model_name,trial_num=30)
    
    dic_path = ''.join(['./model/',model_name,'/trial/trial_',str(trial.number),'.model'])
    args_path = ''.join(['./model/',model_name,'/trial/args_',str(trial.number),'.miarg'])
    
    error_table = model_eval(args_dic=args_path,net_dic=dic_path)
    print(error_table)

def optuna_study(save_dic='./study.pkl',trial_num=5,study_name='study'):
    study = optuna.create_study(direction='minimize',study_name=study_name)
    study.optimize(mpp_obj,n_trials=trial_num)
    with open(save_dic,'wb') as path:
        pickle.dump(study,path)
        path.close()
    
    return study.best_trial

def model_eval(args_dic,net_dic,test_data_path='./data/meta/test_array.npy'):
    args = torch.load(args_dic)
    net_state = torch.load(net_dic)
    
    net = mpp_net(args=args).to(args.device)
    net.load_state_dict(net_state)

    criterion = nn.MSELoss()
    test_item_idx = np.load(test_data_path)
    test_loader = stp_dataloader(test_item_idx=test_item_idx)
    set_file_list = read_set_file()

    error_table = test(net,test_loader,criterion,set_file_list,device=args.device)

    return error_table

if __name__=='__main__':
    main()