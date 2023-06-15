import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class BertClassification(nn.Module):
    def __init__(self, input_dim: int, count_class: int, device, fix: bool = False, backbone: str = "base"):
        super().__init__()
        self.mlp_network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, count_class)
        ).to(device)
        self.network = nn.Linear(input_dim, count_class).to(device)
        self.backbone = BertModel.from_pretrained(backbone).to(device)
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
        self.count_classes = count_class
        self.input_dim = input_dim
        self.fix = fix

    def forward(self, batch):
        tokenized = batch
        # (bsz, L, dim)
        if self.fix:
            with torch.no_grad():
                encoded = self.backbone(tokenized)[0][:, 0, :]
            logits = self.mlp_network(encoded)
        else:
            encoded = self.backbone(tokenized)[0][:, 0, :]
            logits = self.network(encoded)
        return logits
