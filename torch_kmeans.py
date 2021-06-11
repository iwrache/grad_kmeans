import torch
import numpy as np
from kmeans_pytorch import kmeans, kmeans_predict
import torchvision
from collections import Counter
# 导入MNIST数据集

images_data = dict()
labels_data = dict()
data_tmp = torchvision.datasets.MNIST(root='./library', download=True, train=True)
images_data['train'] = data_tmp.data.reshape(-1, 784)
labels_data['train'] = data_tmp.targets
data_tmp = torchvision.datasets.MNIST(root='./library', download=True, train=False)
images_data['test'] = data_tmp.data.reshape(-1, 784)
labels_data['test'] = data_tmp.targets
del data_tmp
# data
data_size, dims, num_clusters = 60000, 784, 15


# kmeans
cluster_ids_x, cluster_centers = kmeans(X=images_data['train'], num_clusters=num_clusters, distance='euclidean',
                                        device=torch.device('cpu'))

cluster_ids_y = kmeans_predict(
    images_data['test'], cluster_centers, 'euclidean')

index = dict()
for i in range(num_clusters):
    index[str(i)] = list(torch.where(cluster_ids_x == i))

labels_map = dict()
for i in range(num_clusters):
    labels_map[str(i)] = Counter(labels_data['train'][index[str(i)]].numpy()).most_common(1)[0][0]

print(labels_map)

count = 0
for i in range(10000):
    if labels_map[str(cluster_ids_y[i].item())] == labels_data['test'][i]:
        count += 1
print('test 10000 images, got ', count, 'images correct')
print('accuracy:', count / 10000)
