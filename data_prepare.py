# 这个文件的目的是将AG_News数据集写到h5里面方便后面进行同步以及与fedml兼容
from datasets import load_dataset
import h5py
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--count-clients", type=int, required=True)
parser.add_argument("--output-file", type=str, required=False, default="partition.h5")
args = parser.parse_args()
pf = h5py.File("/home/Dench991228/fednlp_data/partition_files/agnews_partition.h5", 'r')
print(pf['uniform']['partition_data']['0']['train'][()])
# 原来的分割一共有多少客户端
count_clients = pf['uniform']['n_clients'][()]
# 现在的客户端到原来的客户端的对应表
current_old = {str(current): {"train": [], "test": []} for current in range(args.count_clients)}
old = [i for i in range(count_clients)]
for idx in range(count_clients):
    new_idx = str(idx % args.count_clients)
    train_list = np.array(pf['uniform']['partition_data'][str(idx)]['train'][()])
    print(train_list)
    test_list = np.array(pf['uniform']['partition_data'][str(idx)]['test'][()])
    current_old[new_idx]['train'].extend(train_list)
    current_old[new_idx]['test'].extend(test_list)
output = h5py.File(args.output_file, "w")
output.create_group("uniform")
output['uniform'].create_dataset("n_clients", data=args.count_clients)
output['uniform'].create_group("partition_data")
# 开始写入新的东西
print(current_old.keys())
for idx in range(args.count_clients):
    output['uniform']['partition_data'].create_group(str(idx))
    output['uniform']['partition_data'][str(idx)]['train'] = current_old[str(idx)]['train']
    output['uniform']['partition_data'][str(idx)]['test'] = current_old[str(idx)]['test']
output.close()
output = h5py.File(args.output_file, "r")
print(output.keys())
print(output['uniform'].keys())
print(output['uniform']['n_clients'])
print(output['uniform']['partition_data']['0']['train'][()].shape)
print(output['uniform']['partition_data']['0']['test'][()].shape)