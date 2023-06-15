import argparse

import torch

from model import BertClassification

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", type=str, help="数据文件存放的位置", default="agnews_data.h5")
parser.add_argument("--partition-path", type=str, help="分割文件存放的位置", default="partition.h5")
parser.add_argument("--client-id", type=int, help="当前客户端的编号", required=True)
parser.add_argument("--backbone", type=str, default="bert-base-uncased",
                    choices=["bert-base-uncased", "bert-large-uncased"])
parser.add_argument("--fix", action="store_true")
parser.add_argument("--prompt", action="store_true")
parser.add_argument("--round", type=int, default=5)
args = parser.parse_args()
# 获得模型的维度
encoder_dim = 768 if args.backbone == "bert-base-uncased" else 1024
# 模型初始化
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
f_model = BertClassification(input_dim=encoder_dim, count_class=4, device=DEVICE, fix=args.fix, backbone=args.backbone)
