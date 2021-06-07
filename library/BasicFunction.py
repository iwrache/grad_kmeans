import torch as tc
import pynvml

def get_best_gpu(device='cuda'):
    # isinstance() 函数来判断一个对象是否是一个已知的类型
    if isinstance(device, tc.device):
        return device
    elif device == 'cuda':
        pynvml.nvmlInit()   # 初始化
        num_gpu = pynvml.nvmlDeviceGetCount()   # 有几块gpu
        memory_gpu = tc.zeros(num_gpu)
        for index in range(num_gpu):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_gpu[index] = memory_info.free
        # 得到各个gpu中的空闲内存
        max_gpu = int(tc.sort(memory_gpu, )[1][-1])  # 得到空闲最大的那个gpu的编号
        return tc.device('cuda:' + str(max_gpu))
    elif device == 'cpu':
        return tc.device('cpu')