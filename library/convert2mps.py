import numpy as np
import torch as tc
import time
import torchvision
from library import MPSclass
# from library import TNclass
from library import BasicFunction
from library import Para
import copy
import math

class MachineLearning(MPSclass.MPS):

    def __init__(self, tensor_data, tensor_input, para=Para.gtnc(), device='cuda', dtype=tc.float64):
        # initialize parameters
        print('len:', len(tensor_input))
        MPSclass.MPS.__init__(self)

        self.para = copy.deepcopy(para)
        self.tensor_input = tensor_input

        self.index = dict()
        self.labels_data = dict()
        self.update_info = dict()
        self.data_info = dict()
        self.tmp = {}
        self.environment_zoom = tuple()


        self.device = BasicFunction.get_best_gpu(device=device)

        self.dtype = dtype
        self.tensor_data = tensor_data
        self.tensor_info = dict()

    def norm_mps(self): # 初始化一开始的mps
        if self.para['dataset'] == 'mnist':
            self.data_info['n_feature'] = self.tensor_info['n_length'] = 784
            self.tensor_info['regular_center'] = 'unknown'
            self.tensor_info['cutoff'] = self.para['mps_cutoff']

        self.mps_regularization(-1)
        self.mps_regularization(0)
        return self.tensor_data

    def prepare_start_learning(self):
        self.generate_update_info()
        # self.tensor_input = self.feature_map(self.images_data['dealt_input'])   # 6w*784*2, :,:,0是sin，:,:,1是cos
        self.environment_left = list(range(self.data_info['n_feature']))  # 784
        
        self.environment_right = list(range(self.data_info['n_feature']))
        
        self.environment_zoom = dict()
        self.initialize_environment()
        #if self.update_info['loops_learned'] != 0:
        print('load mps trained ' + str(self.update_info['loops_learned']) + ' loops')

    # 要更新的一些信息
    def generate_update_info(self):
        self.data_info['labels'] = tuple(range(10))
        self.data_info['n_training'] = self.tensor_input.shape[0]
        self.data_info['n_feature'] = self.tensor_input.shape[1]
        self.data_info['n_sample'] = dict()
        self.data_info['n_sample']['train'] = self.tensor_input.shape[0]
        self.update_info['update_direction'] = 1
        self.update_info['loops_learned'] = 0
        self.update_info['cost_function_loops'] = list()
        self.update_info['cost_time_cpu'] = list()
        self.update_info['cost_time_wall'] = list()
        self.update_info['step'] = self.para['update_step']
        self.update_info['is_converged'] = 'untrained'
        #self.update_info['update_mode'] = self.para['update_mode']
        self.tensor_info['regular_center'] = 'unknown'
        self.tensor_info['n_length'] = 28 * 28
        self.tensor_info['cutoff'] = self.para['mps_cutoff']
        self.update_info['update_position'] = 0

    # 初始化环境
    def initialize_environment(self):
        self.environment_zoom['left'] = tc.zeros(
            (self.data_info['n_feature'], self.data_info['n_training']),
            device=self.device, dtype=self.dtype)  # 784*(6k左右)
        self.environment_zoom['right'] = tc.zeros(
            (self.data_info['n_feature'], self.data_info['n_training']),
            device=self.device, dtype=self.dtype)

        ii = 0
        self.environment_left[ii] = tc.ones(self.data_info['n_training'], device=self.device, dtype=self.dtype).unsqueeze(-1)
        ii = self.data_info['n_feature'] - 1
        self.environment_right[ii] = tc.ones(self.data_info['n_training'], device=self.device, dtype=self.dtype).unsqueeze(-1)
        # for ii in range(self.tensor_info['n_length'] - 1):
        #     self.calculate_environment_next(ii + 1)
        for ii in range(self.data_info['n_feature'] - 1, 0, -1):
            self.calculate_environment_forward(ii - 1)

        # print('equal:', (np.exp(self.environment_zoom['right'][0, :]) * self.environment_right[0]).shape)

    def calculate_environment_forward(self, environment_index):

        self.environment_right[environment_index] = tc.einsum(
            'nv,ivj,nj->ni',
            self.tensor_input[:, environment_index + 1, :],
            self.tensor_data[environment_index + 1],
            self.environment_right[
                environment_index + 1])
        # input shape:6k*784*2, data shape:(type:list,第i项type为bond*physical_bond*bond)，environment_right shape:6k*bond
        # 得到的结果为6k*left_bond
        tmp_norm = (self.environment_right[environment_index]).norm(dim=1)  # 计算该矩阵每一列的范数，返回的是长为6k的Tensor

        self.environment_zoom['right'][environment_index, :] = \
            self.environment_zoom['right'][environment_index + 1, :] + tc.log(tmp_norm)  # 存放每一步归一化后多出来的范数值
        self.environment_right[environment_index] = tc.einsum(
            'ij,i->ij', self.environment_right[environment_index], 1 / tmp_norm)  # 归一化，使其内积等于1
        # print('right_env norm:', tc.norm(self.environment_right[environment_index]))

    def calculate_environment_next(self, environment_index):
        self.environment_left[environment_index] = tc.einsum(
            'ni,ivj,nv->nj',
            self.environment_left[environment_index - 1],
            self.tensor_data[environment_index - 1],
            self.tensor_input[:, environment_index - 1, :])

        tmp_norm = self.environment_left[environment_index].norm(dim=1)
        self.environment_zoom['left'][environment_index, :] = \
            self.environment_zoom['left'][environment_index - 1, :] + tc.log(tmp_norm)
        self.environment_left[environment_index] = tc.einsum(
            'ij,i->ij', self.environment_left[environment_index], 1 / tmp_norm)
        # print('left_env norm:', tc.norm(self.environment_left[environment_index]))

