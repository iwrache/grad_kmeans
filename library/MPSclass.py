import numpy
import torch
import operator
from library import TNclass
import library.wheel_function as wf
import copy

class MPS(TNclass.TensorNetwork):
    def __init__(self):
        # Prepare parameters

        TNclass.TensorNetwork.__init__(self)

    # 将mps变成中心点在regular_center的正交形式
    def mps_regularization(self, regular_center):
        if regular_center == -1:
            regular_center = self.data_info['n_feature'] - 1
        # 如果regular_center等于-1，表明想把正则中心移动至最后一个点

        if self.tensor_info['regular_center'] == 'unknown':
            self.tensor_info['regular_center'] = 0
            while self.tensor_info['regular_center'] < self.tensor_info['n_length']-1:
                self.move_regular_center2next()
        # 如果不知道正则中心，就先将tensor右规范化
        while self.tensor_info['regular_center'] < regular_center:
            self.move_regular_center2next()
        while self.tensor_info['regular_center'] > regular_center:
            self.move_regular_center2forward()

    def move_regular_center2next(self):
        # 注意torch中是u@s@v.T = matrix
        # self.reverse_mps()
        # self.move_regular_center2forward()
        # self.reverse_mps()
        tensor_index = self.tensor_info['regular_center']
        u, s, v = wf.tensor_svd(self.tensor_data[tensor_index], index_right=2)
        s /= s.norm()
        s = s[s > self.tensor_info['cutoff']]
        dimension_middle = min([len(s), self.para['virtual_bond_limitation']])
        u = u[:, 0:dimension_middle].reshape(-1, dimension_middle)
        s = s[0:dimension_middle]
        v = v[:, 0:dimension_middle].reshape(-1, dimension_middle)
        self.tensor_data[tensor_index] = u.reshape(
            self.tensor_data[tensor_index].shape[0],
            self.tensor_data[tensor_index].shape[1], dimension_middle)

        if self.para['training_label'] == 1:
            print('index:', tensor_index, (torch.diag(s)).mm(v.t()).shape, self.tensor_data[tensor_index+1].shape)

        self.tensor_data[tensor_index + 1] = torch.einsum(
            'ij,jkl->ikl',
            [(torch.diag(s)).mm(v.t()), self.tensor_data[tensor_index+1]])
        self.tensor_info['regular_center'] += 1

    def move_regular_center2forward(self):
        # self.reverse_mps()
        # self.move_regular_center2next()
        # self.reverse_mps()
        tensor_index = self.tensor_info['regular_center']
        u, s, v = wf.tensor_svd(self.tensor_data[tensor_index], index_left=0)

        s /= s.norm()
        s = s[s > self.tensor_info['cutoff']]
        dimension_middle = min([len(s), self.para['virtual_bond_limitation']])
        u = u[:, 0:dimension_middle].reshape(-1, dimension_middle)
        s = s[0:dimension_middle]
        v = v[:, 0:dimension_middle].reshape(-1, dimension_middle)
        self.tensor_data[tensor_index] = v.t().reshape(
            dimension_middle, self.tensor_data[tensor_index].shape[1],
            self.tensor_data[tensor_index].shape[2])
        self.tensor_data[tensor_index - 1] = torch.einsum(
            'ijk,kl->ijl',
            self.tensor_data[tensor_index - 1], u.mm(torch.diag(s)))
        self.tensor_info['regular_center'] -= 1

    def mps_inner_product(self):

        inn = torch.ones((1, 1), dtype=self.dtype, device=self.device)
        for i in range(len(self.tensor_data)):
            inn = torch.einsum('ij,jkm,ikl->lm', inn, self.tensor_data[i], self.tensor_data[i])
        print('tensor data inner product equals to ', inn.squeeze())

