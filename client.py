import logging

from flwr.common import logger
import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from loader import load_data
import argparse
from model import BertClassification
from torch.utils.data import DataLoader
from options import args, DEVICE, f_model

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
logger.logger.setLevel(logging.INFO)
logging.basicConfig(filename=f"client{args.client_id}.logfile", format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model: torch.nn.Module, train_loader: DataLoader, test_loader: DataLoader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        logger.logger.info("start training")
        train(self.model, self.train_loader, epochs=1)
        loss, accuracy = test(self.model, self.test_loader)
        logger.logger.info(f"evaluate result on local data and local model: acc {accuracy}")
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.test_loader)
        logger.logger.info(f"evaluate result on local data: acc {accuracy}")
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}


def train(net, trainloader: DataLoader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, betas=(0.99, 0.999))
    for _ in range(epochs):
        for token_ids, labels in tqdm(trainloader):
            optimizer.zero_grad()
            logits = net(token_ids.to(DEVICE))
            criterion(logits, labels.to(DEVICE)).backward()
            optimizer.step()
            pred = torch.argmax(logits, dim=-1)
            correct = (pred == labels)
            correct = torch.sum(correct.float()).item()
            total = labels.shape[0]
            logger.logger.info(pred)
            logger.logger.info(correct*1.0/total)
            logger.logger.info(labels)


def test(net, testloader: DataLoader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    count = 0
    with torch.no_grad():
        for token_ids, labels in tqdm(testloader):
            outputs = net(token_ids.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            correct += (torch.argmax(outputs, dim=-1) == labels).float().sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


# 加载数据
t_loader, e_loader, s_loader, count_train, count_eval = load_data(client_idx=args.client_id,
                                                                  data_file=args.data_path,
                                                                  partition_file=args.partition_path,
                                                                  batch_size=4,
                                                                  tokenizer_name=args.backbone)

# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(model=f_model, train_loader=t_loader, test_loader=e_loader),
    grpc_max_message_length=1024 * 1024 * 1024
)
