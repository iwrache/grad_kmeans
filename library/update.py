import numpy as np
import torch as tc
import copy
from library import convert2mps
from library import Para
import time

class GTNC(convert2mps.MachineLearning):
    def __init__(self, tensor_data, tensor_input, para=Para.gtnc(), device='cuda'):

        convert2mps.MachineLearning.__init__(self, tensor_data, tensor_input, para, device)
        self.inner_product = dict()
        self.data_mapped = dict()
        # self.right_label = dict()
        self.accuracy = dict()
        self.test_info = dict()


    # 打印程序起始时间
    def print_program_info(self, mode='start'):
        if mode == 'start':
            print('This program starts at ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        elif mode == 'end':
            print('This program ends at ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


    # 开始学习，计算一开始的cost function->一圈一圈学习知道收敛或者学习次数达到上限,这个函数只训练一个类
    def start_learning(self, learning_loops=3):
        self.prepare_start_learning()
        if self.update_info['is_converged'] == 'untrained':
            self.update_info['is_converged'] = False
        if self.update_info['loops_learned'] == 0:
            self.calculate_cost_function()
            self.update_info['cost_function_loops'].append(self.update_info['cost_function'])
            self.learning_rate()
            print('Initializing ... cost function = ' + str(self.update_info['cost_function']))
        if not self.update_info['is_converged']:
            print('start to learn to ' + str(learning_loops) + ' loops')
        while (self.update_info['loops_learned'] < learning_loops) and not (self.update_info['is_converged']):
            self.update_one_loop()
            self.is_converge()

        if self.update_info['is_converged']:
            self.print_converge_info()
        else:
            print('Training end, cost function = ' + str(self.update_info['cost_function']) + ', do not converge.')
        return self.tensor_data

    # 打印收敛的时候信息
    def print_converge_info(self):
        print('cost function is converged at ' + str(self.update_info['cost_function'])
              + '. Program terminates')
        print('Train ' + str(self.update_info['loops_learned']) + ' loops')

    # 一个一个site更新张量，做一个来回
    def update_one_loop(self):

        if self.tensor_info['regular_center'] != 0:
            self.mps_regularization(-1)
            self.mps_regularization(0)

        self.update_info['update_position'] = self.tensor_info['regular_center']
        self.update_info['update_direction'] = +1
        while self.tensor_info['regular_center'] < self.tensor_info['n_length'] - 1:
            self.update_mps_once()
            # self.mps_inner_product()
            self.mps_regularization(self.update_info['update_position'] + self.update_info['update_direction'])
            self.update_info['update_position'] = self.tensor_info['regular_center']
            self.calculate_environment_next(self.update_info['update_position'])
        self.update_info['update_direction'] = -1
        while self.tensor_info['regular_center'] > 0:
            self.update_mps_once()
            self.mps_regularization(self.update_info['update_position'] + self.update_info['update_direction'])
            self.update_info['update_position'] = self.tensor_info['regular_center']
            self.calculate_environment_forward(self.update_info['update_position'])
        # print('loop:')
        # self.mps_inner_product()
        self.tensor_data[0] /= (self.tensor_data[0]).norm()
        self.calculate_cost_function()  # 每次做完一个loop求cost,所以这时候的归一化中心一定在0的位置
        print('cost function = ' + str(self.update_info['cost_function'])
              + ' at ' + str(self.update_info['loops_learned'] + 1) + ' loops.')
        # self.mps_inner_product()
        self.update_info['cost_function_loops'].append(self.update_info['cost_function'])
        self.update_info['loops_learned'] += 1

    def update_mps_once(self): # cost function不是直接的NLL函数，前面还有个ln(Z')/ln(Z)，看原文就知道了
        # Calculate gradient
        tmp_index1 = self.tensor_info['regular_center']
        tmp_tensor_current = self.tensor_data[tmp_index1]
        tmp_tensor1 = tc.einsum(
            'ni,nv,nj->nivj',
            self.environment_left[tmp_index1],
            self.tensor_input[:, tmp_index1, :],
            self.environment_right[tmp_index1]).reshape(self.data_info['n_training'], -1)   # 这是内积挖掉current tensor的部分，reshape为6k*(left * physical_bond * right)
        tmp_inner_product = (tmp_tensor1.mm(tmp_tensor_current.view(-1, 1))).t()    # 得到mps与6k个数的内积,shape为1*6k
        tmp_tensor1 = ((1/tmp_inner_product).mm(tmp_tensor1)).reshape(tmp_tensor_current.shape)     # 偏f偏A里sum的值，得到的shape为left * physical_bond *right

        self.tmp['gradient'] = 2 * (
                (tmp_tensor_current / (tmp_tensor_current.norm() ** 2))
                - tmp_tensor1 / self.data_info['n_training'])

        tmp_tensor_norm = self.tmp['gradient'].norm()

        # Update MPS

        tmp_tensor_current -= self.update_info['step'] * self.tmp['gradient'] / (
                tmp_tensor_norm + self.para['tensor_acc'])
        self.tensor_data[self.update_info['update_position']] = tmp_tensor_current

    # 计算cost function，这里np.log(self.data_info['n_training'])设为负的结果比正的高零点几
    def calculate_cost_function(self):
        if self.update_info['update_position'] != 0:
            print('go check your code')
        tmp_matrix = self.tensor_input[:, 0, :]@self.tensor_data[0][0, :, :]    # 6k*2 * 2*bond(2)
        tmp_inner_product = tc.einsum('ij,ij->i', tmp_matrix, self.environment_right[0]).cpu()  # 与源代码结果一样，要是我我就这么写
        # tmp_inner_product = ((tmp_matrix.mul(self.environment_right[0])).sum(1)).cpu()
        # mul:对两个张量进行逐元素乘法, .sum(1):将每一行的数相加，得到一个长为6k的tensor

        self.update_info['cost_function'] = 2 * np.log((
            self.tensor_data[0]).norm().cpu()) - np.log(self.data_info['n_training']) - 2 * sum(
            self.environment_zoom['right'][0, :].cpu() + np.log(abs(tmp_inner_product))) / self.data_info['n_training']
        # cost function和原来的公式差别不大，第一项是几乎为0，可以忽略，第二项是对每个类别加了一个常数(该类别中图片数量)，第三项是公式原本的东西。第二项去掉不去掉影响不大，准确率就差一点点

    # 判断两次cost function的比率是否收敛
    def is_converge(self):
        loops_learned = self.update_info['loops_learned']
        cost_function_loops = self.update_info['cost_function_loops']
        # print('cost_function_loop:', cost_function_loops)
        self.update_info['is_converged'] = bool(
            ((cost_function_loops[loops_learned - 1] - cost_function_loops[loops_learned]) /
             abs(cost_function_loops[loops_learned - 1])) < self.para['converge_accuracy'])

    # 调整学习率
    def learning_rate(self):
        self.update_info['step'] = self.para['update_step']
        if 50 <= self.update_info['cost_function']:
            self.update_info['step'] /= self.para['step_decay_rate']
        elif 30 <= self.update_info['cost_function']:
            self.update_info['step'] /= self.para['step_decay_rate']**2
        elif 10 <= self.update_info['cost_function']:
            self.update_info['step'] /= self.para['step_decay_rate']**3

