import torch as tc
import torchvision
import copy
import numpy as np
from collections import Counter
from library import update
from library import Para
from library import convert2mps

class K_means():
    def __init__(self, k=15, para=Para.gtnc(), dtype=tc.float64, device='cpu'):

        self.tensor_data = dict()
        self.device = device
        self.k = k
        self.max_iter_time = 2
        self.dtype = dtype
        self.convergence = False
        self.para = copy.deepcopy(para)
        self.images_data = dict()
        self.labels_data = dict()
        self.index = dict()
        self.data_info = dict()
        if self.para['dataset'] == 'mnist':
            self.n_feature = 28**2
        self.inner_product = dict()
        for ii in self.para['data_type']:
            self.inner_product[ii] = dict()
        self.init_mps()
        self.last_tensor_data = copy.deepcopy(self.tensor_data)


        for label in range(self.k):
            tmp = convert2mps.MachineLearning(self.tensor_data[str(label)], tensor_input=dict(), para=self.para, device='cpu')
            self.tensor_data[str(label)] = tmp.norm_mps()
            # 将每一个随机生成的mps归一化
        self.initialize_dataset()
        self.iter()


    # 随机生成一开始的mps，但这里没有归一化
    def init_mps(self):
        for label in range(self.k):
            self.tensor_data[str(label)] = list()

            for ii in range(self.n_feature):
                self.tensor_data[str(label)].append(tc.rand(
                    self.para['virtual_bond_limitation'],
                    self.para['physical_bond'],
                    self.para['virtual_bond_limitation'],
                    dtype=self.dtype))

            self.tensor_data[str(label)][0] = tc.rand(
                1, self.para['physical_bond'],
                self.para['virtual_bond_limitation'],
                dtype=self.dtype)

            self.tensor_data[str(label)][-1] = tc.rand(
                self.para['virtual_bond_limitation'],
                self.para['physical_bond'], 1,
                dtype=self.dtype)

    # 初始化数据集，将数据集中每个元素置于0-1之间,然后将cos和sin作用上去
    def initialize_dataset(self):
        self.load_dataset() # 加载mnist数据集
        # self.data_info['labels'] = tuple(range(10))
        self.calculate_dataset_info()   # 得到mnist数据集的一些信息
        self.images_data['dealt_input'] = self.deal_data(self.images_data['train'])  # 将数据集中每个数缩小到0-1之间
        # mnist数据集就是6w左右*784，不过将其每个数都缩小到了0-1之间
        self.data_info['n_training'], self.data_info['n_feature'] = self.images_data['dealt_input'].shape
        # print('6w:', self.data_info['n_training'], '784:', self.data_info['n_feature'])
        self.tensor_input = self.feature_map(self.images_data['dealt_input'])  # 6w*784*2, :,:,0是sin，:,:,1是cos
        self.arrange_data()

    # 迭代更新mps
    def iter(self):
        iter_time = 0
        while iter_time < self.max_iter_time:
            for label in range(self.k):
                if self.images_data['input'][str(label)].shape[0] != 0:
                    print('shape0:', self.images_data['input'][str(label)].shape[0])
                    tmp_mps = update.GTNC(self.tensor_data[str(label)], tc.from_numpy(self.images_data['input'][str(label)]), para=self.para, device=self.device)
                    self.tensor_data[str(label)] = tmp_mps.start_learning()
            self.judge_convergence()
            if self.convergence == True:
                print('trained ', iter_time, 'times, the data is convergence!')
                break
            else:
                iter_time += 1
                self.arrange_data()

    # 判断是否收敛
    def judge_convergence(self):
        total_error = 0
        for label in range(self.k):
            inn = tc.ones((1, 1), dtype=self.dtype, device=self.device)
            for i in range(len(self.tensor_data[str(label)])):
                inn = tc.einsum('ij,jkm,ikl->lm', inn, self.last_tensor_data[str(label)][i], self.tensor_data[str(label)][i])
            total_error += abs(inn.squeeze())
        if total_error < 1e-10:
            self.convergence = True
        else:
            self.last_tensor_data = copy.deepcopy(self.tensor_data)

    # 看测试集准确率
    def accuracy(self):
        labels_map = dict()
        for label in range(self.k):
            if len(self.index['train']['divided'][label]) == 0:
                raise AssertionError('这一类没有数据')
            else:
                labels_map[str(label)] = Counter(self.labels_data['train'][
                                                     self.index['train']['divided'][label]]).most_common(1)[0][0]


        self.images_data['test_mapped'] = self.feature_map(self.deal_data(self.images_data['test']))
        test_images = self.images_data['test_mapped'].shape[0]
        inner_storage = np.zeros((test_images, self.k))

        for label in range(self.k):
            tmp_inner_product = tc.ones((test_images, 1), device=self.device, dtype=self.dtype)
            for ii in range(self.data_info['n_feature']):
                tmp_inner_product = tc.einsum('ni,ivj,nv->nj', tmp_inner_product, self.tensor_data[str(label)][ii],
                                              self.images_data['test_mapped'][:, ii, :])

            inner_storage[:, label] = tmp_inner_product.squeeze()

        test_class = [np.argmax(c) for c in inner_storage]

        count = 0
        for ii in range(test_images):
            if labels_map[str(test_class[ii])] == self.labels_data['test'][ii]:
                count += 1

        print(
            'test ' + str(test_images)
            + ' images, got ' + str(count) + ' right images')
        print('the accuracy is ' + str(count/test_images))


    # 加载数据集
    def load_dataset(self):
        print('path:', self.para['path_dataset'])
        if self.para['dataset'] == 'mnist':
            # 当train=True时表明是训练集，否则表示测试集
            data_tmp = torchvision.datasets.MNIST(root=self.para['path_dataset'], download=True, train=True)
            self.images_data['train'] = data_tmp.data.numpy().reshape(-1, 784)  # shape:60000*784
            self.labels_data['train'] = data_tmp.targets.numpy()    # 默认类型为tc.Tensor。shape:6w

            data_tmp = torchvision.datasets.MNIST(root=self.para['path_dataset'], download=True, train=False)
            self.images_data['test'] = data_tmp.data.numpy().reshape(-1, 784)
            self.labels_data['test'] = data_tmp.targets.numpy()
            del data_tmp

    # 将数据集中每个元素的值置于0-1之间
    def deal_data(self, image_data):
        tmp_image_data = image_data.copy()
        tmp_image_data = np.double(tmp_image_data / tmp_image_data.max())
        return tmp_image_data

    # 将每个数映射到cos 和 sin
    def feature_map(self, image_data_mapping):
        image_data_mapping = image_data_mapping * self.para['theta']
        image_data_mapping = tc.tensor(image_data_mapping, device=self.device, dtype=self.dtype)
        image_data_mapped = tc.zeros((image_data_mapping.shape + (2,)), device=self.device, dtype=self.dtype)

        for ii in range(2):
            image_data_mapped[:, :, ii] = (tc.sin(image_data_mapping) ** (1 - ii)) * (tc.cos(image_data_mapping) ** ii)
        return image_data_mapped

    # 得到数据集和测试集的样本数量和每个样本的长度
    def calculate_dataset_info(self):
        self.data_info['n_sample'] = dict()
        for data_type in self.para['data_type']:
            self.data_info['n_sample'][data_type] = self.images_data[data_type].shape[0]
        for data_type in self.para['data_type']:
            self.index[data_type] = dict()
            self.index[data_type]['origin'] = np.arange(self.data_info['n_sample'][data_type])

    def arrange_data(self):
        self.divide_data(data_type='train')
        self.images_data['input'] = dict()
        for label in range(self.k):
            self.images_data['input'][str(label)] = np.array((self.tensor_input[
                self.index['train']['divided'][label]]))
            # shape:每个label的数量*784
            # print('shape:', self.images_data['input'][str(label)].shape)

    def divide_data(self, data_type):
        tmp = self.calculate_inner_product(data_type)

        self.index[data_type]['divided'] = dict()

        for label in range(self.k):
            self.index[data_type]['divided'][label] = np.where(label == tmp)[0]
            print('label:', len(self.index[data_type]['divided'][label]))

    # 和下面1一起，都是计算内积的，得到的结果的shape为1w
    def calculate_inner_product(self, data_type):

        storage = np.zeros((self.data_info['n_training'], self.k))
        self.inner_product[data_type] = dict()
        for label in range(self.k):
             storage[:, label] = self.calculate_inner_product1(label).cpu().numpy()
        tmp_class = [np.argmax(c) for c in storage]
        cost = sum([np.max(c) for c in storage])
        print('cost:', cost)
        return np.array(tmp_class)
        # print('tmp_class:', tmp_class)
        # self.inner_product[data_type][str(label)]

    def calculate_inner_product1(self, label):
        n_images = self.tensor_input.shape[0]
        # print('n images:', n_images)
        tmp_inner_product = tc.ones((n_images, 1), device=self.device, dtype=self.dtype)
        for ii in range(self.data_info['n_feature']):
            tmp_inner_product = tc.einsum('ni,ivj,nv->nj', tmp_inner_product, self.tensor_data[str(label)][ii],
                                          self.tensor_input[:, ii, :])
        # tmp_inner_product = tmp_inner_product.reshape(-1)
        tmp_inner_product = tmp_inner_product.squeeze()
        return tmp_inner_product

if __name__=="__main__":
    tmp = K_means()
    tmp.accuracy()
