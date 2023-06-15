import h5py
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co
from typing import List, Tuple
from transformers import BertTokenizer


class MyDataset(Dataset):

    def __init__(self, instances: List[str], labels: List[int]):
        self.instances = instances
        self.labels = labels

    def __getitem__(self, index) -> T_co:
        return self.instances[index], self.labels[index]

    def __len__(self):
        return len(self.instances)


def load_all_instances(data_file: h5py.File):
    f = data_file
    instances = {}
    labels = {}
    indices = [int(k) for k in f['X'].keys()]
    for index in indices:
        instance = f['X'][str(index)][()].decode("UTF-8")
        label = f['Y'][str(index)][()].decode("UTF-8")
        instances[index] = instance
        labels[index] = label
    return instances, labels


def load_data(client_idx: int, data_file: str, partition_file: str, batch_size: int, tokenizer_name: str):
    """
    加载训练/测试数据
    :param client_idx: 客户端的编号，和加载的数据有关
    :param data_file: 存放数据的文件
    :param partition_file: 存放split的文件
    :param batch_size: 加载数据的时候的bsz
    :param tokenizer_name: 进行tokenize的
    :return: train_loader, test_loader
    """
    # 原始数据的文件
    f = h5py.File(data_file, 'r')
    # 分割文件
    pf = h5py.File(partition_file, 'r')
    # 先加载全部的输入样本和labels
    instances, labels = load_all_instances(f)
    # 训练和测试的下标，以及相关的样本
    train_indices = list(pf['uniform']['partition_data'][str(client_idx)]['train'][()])
    train_instances = []
    train_labels = []
    for t_idx in train_indices:
        train_instances.append(instances[t_idx])
        train_labels.append(int(labels[t_idx]) - 1)
    test_indices = pf['uniform']['partition_data'][str(client_idx)]['test'][()]
    test_instances = []
    test_labels = []
    for t_idx in test_indices:
        test_instances.append(instances[t_idx])
        test_labels.append(int(labels[t_idx]) - 1)
    # 服务器端的测试集
    server_indices = []
    for idx in pf['uniform']['partition_data'].keys():
        indices = list(pf['uniform']['partition_data'][idx]['test'][()])
        server_indices.extend(indices)
    server_instances = []
    server_labels = []
    for s_idx in server_indices:
        server_instances.append(instances[s_idx])
        server_labels.append(int(labels[s_idx])-1)

    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    def sentence_to_tensor(batch: List):
        input_sentences = [item[0] for item in batch]
        input_token_ids = tokenizer(input_sentences, truncation=True, max_length=512, padding="longest")['input_ids']
        input_labels = [item[1] for item in batch]
        return torch.LongTensor(input_token_ids), torch.LongTensor(input_labels)

    train_dataset = MyDataset(train_instances, train_labels)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size,
                              num_workers=4, collate_fn=sentence_to_tensor)
    test_dataset = MyDataset(test_instances, test_labels)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size * 4,
                             num_workers=4, collate_fn=sentence_to_tensor)
    server_dataset = MyDataset(server_instances, server_labels)
    server_loader = DataLoader(dataset=server_dataset, shuffle=False, batch_size=batch_size*4, num_workers=4, collate_fn=sentence_to_tensor)
    return train_loader, test_loader, server_loader, len(train_dataset), len(test_dataset)
