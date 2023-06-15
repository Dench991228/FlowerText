import json
from typing import List, Tuple, Dict, Optional
from collections import OrderedDict
from torch.utils.data import DataLoader

import flwr as fl
import torch.nn
from flwr.common import Metrics
from flwr.common.typing import Parameters
import logging
from options import args, f_model, DEVICE

from tqdm import tqdm
from loader import load_data
import numpy

logging.basicConfig(filename="server.logfile", format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    logging.info(json.dumps({"accuracy": sum(accuracies) / sum(examples)}))
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def get_evaluation_fn(model: torch.nn.Module, loader: DataLoader):
    # 准备测试用的数据集
    # 初始化测试用的模型
    # 最终返回的函数
    best_acc = 0.0
    def evaluate(
            server_round: int,
            parameters: fl.common.NDArrays,
            config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        ms, un = model.load_state_dict(state_dict, strict=True)
        logging.info(f"{ms}, unexpected: {un}")
        criterion = torch.nn.CrossEntropyLoss()
        count_correct = 0
        count_total = 0
        total_loss = 0.0
        with torch.no_grad():
            for token_ids, labels in tqdm(loader):
                labels = labels.to(DEVICE)
                token_ids = token_ids.to(DEVICE)
                logit = model(token_ids)
                loss = criterion(logit, labels)
                pred = torch.argmax(logit, dim=-1)
                correct = (pred == labels)
                correct = torch.sum(correct.float()).item()
                total = labels.shape[0]
                count_correct += correct
                count_total += total
                total_loss += loss * 1.0 / len(loader)
        acc = count_correct * 1.0 / count_total
        logging.info(f"round {server_round}: Test acc is {acc}")
        # 保存模型
        torch.save(model.state_dict(), f"{args.backbone}{'_f' if args.fix else ''}.ckpt")
        return total_loss, {"accuracy": count_correct * 1.0 / count_total}

    return evaluate


t_loader, e_loader, s_loader, count_train, count_eval = load_data(client_idx=args.client_id,
                                                                  data_file=args.data_path,
                                                                  partition_file=args.partition_path,
                                                                  batch_size=16,
                                                                  tokenizer_name=args.backbone)

# Define strategy
if args.strategy == "avg":
    strategy = fl.server.strategy.FedAvg(
        evaluate_fn=get_evaluation_fn(f_model, s_loader),
        evaluate_metrics_aggregation_fn=weighted_average,
        min_fit_clients=3,
        min_available_clients=3,
        min_evaluate_clients=3
    )
else:
    strategy = fl.server.strategy.FedAdam(
        evaluate_fn=get_evaluation_fn(f_model, s_loader),
        evaluate_metrics_aggregation_fn=weighted_average,
        min_fit_clients=3,
        min_available_clients=3,
        min_evaluate_clients=3,
        initial_parameters=fl.common.ndarrays_to_parameters([val.cpu().numpy() for _, val in f_model.state_dict().items()]),
        tau=0.1,
        eta_l=1e-1,
        eta=1
    )

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=args.round),
    strategy=strategy,
    grpc_max_message_length=1024 * 1024 * 1024
)
