import numpy
import torch

def tensor_svd(tmp_tensor, index_left='none', index_right='none'):
    tmp_shape = numpy.array(tmp_tensor.shape)
    tmp_index = numpy.arange(len(tmp_tensor.shape))
    if index_left == 'none':
        index_right = tmp_index[index_right].flatten()
        index_left = numpy.setdiff1d(tmp_index, index_right)
    elif index_right == 'none':
        index_left = tmp_index[index_left].flatten()
        index_right = numpy.setdiff1d(tmp_index, index_left)
    index_right = numpy.array(index_right).flatten()
    index_left = numpy.array(index_left).flatten()
    tmp_tensor = tmp_tensor.permute(tuple(numpy.concatenate([index_left, index_right])))
    # numpy提供了numpy.concatenate((a1,a2,...), axis=0)函数。能够一次完成多个数组的拼接。其中a1,a2,...是数组类型的参数
    tmp_tensor = tmp_tensor.reshape(tmp_shape[index_left].prod(), tmp_shape[index_right].prod())
    # prod: 计算所有元素的乘积
    u, l, v = torch.svd(tmp_tensor)
    return u, l, v

def tensor_contract(a, b, index):
    ndim_a = numpy.array(a.shape)
    ndim_b = numpy.array(b.shape)
    order_a = numpy.arange(len(ndim_a))
    order_b = numpy.arange(len(ndim_b))
    order_a_contract = numpy.array(order_a[index[0]]).flatten()
    # 假设元素为0,不见flatten则为array(0),加上flatten则为array([0])
    order_b_contract = numpy.array(order_b[index[1]]).flatten()
    order_a_hold = numpy.setdiff1d(order_a, order_a_contract)
    # setdiff1d(ar1, ar2, assume_unique=False),返回在ar1中但不在ar2中的已排序的唯一值，可以从最后看出返回的值从小到大排序，并且唯一
    # assume_unique = True时，可以看出把在a中的但是不在b中的元素按a中的顺序排序，并且不合并重复的元素，即假定输入数组也是唯一的，因此相比于False提升了运算速度。
    order_b_hold = numpy.setdiff1d(order_b, order_b_contract)
    hold_shape_a = ndim_a[order_a_hold].flatten()
    hold_shape_b = ndim_b[order_b_hold].flatten()
    return numpy.dot(
        a.transpose(numpy.concatenate([order_a_hold, order_a_contract])).reshape(hold_shape_a.prod(), -1),
        b.transpose(numpy.concatenate([order_b_contract, order_b_hold])).reshape(-1, hold_shape_b.prod()))\
        .reshape(numpy.concatenate([hold_shape_a, hold_shape_b]))
    # np.dot 等于@
    # np.prod()函数用来计算所有元素的乘积，对于有多个维度的数组可以指定轴，如axis=1指定计算每一行的乘积
    # numpy提供了numpy.concatenate((a1,a2,...), axis=0)函数。能够一次完成多个数组的拼接。
