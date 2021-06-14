import torch as tc
import numpy as np

def gtnc(para = dict()):
    para['dataset'] = 'mnist'
    para['path_dataset'] = './dataset/'
    para['data_type'] = ['train', 'test']
    para['save_data_path'] = './data_trained/'
    para['dtype'] = tc.float64
    para['device'] = 'cuda'
    para['physical_bond'] = 2
    para['virtual_bond_limitation'] = 64
    para['mps_cutoff'] = -1
    para['theta'] = (np.pi / 2)
    para['training_label'] = [[i] for i in range(10)]
    para['converge_accuracy'] = 1e-2
    para['mps_cutoff'] = -1
    para['step_accuracy'] = 5e-3
    para['step_decay_rate'] = 5
    para['tensor_acc'] = 1e-8
    para['update_step'] = 2e-1

    return para
