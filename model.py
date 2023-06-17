import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class PromptModule(nn.Module):
    def __init__(self, encoder_dim: int, device: str, embed: nn.Module):
        super().__init__()
        self.dim = encoder_dim
        self.prompt = torch.zeros((1, encoder_dim)).to(device)
        self.prompt = nn.Parameter(self.prompt)
        nn.init.uniform_(self.prompt, -0.1, 0.1)
        self.base = embed

    def forward(self, token_ids):
        """
        输入的token_ids是单词id的序列
        """
        input_embeddings = self.base(token_ids)
        expanded_prompts = self.prompt.repeat(token_ids.shape[0], 1)
        return torch.cat([expanded_prompts, input_embeddings], 1)


class BertClassification(nn.Module):
    def __init__(self, input_dim: int, count_class: int, device, fix: bool = False, backbone: str = "base",
                 prompt: bool = False):
        super().__init__()
        self.mlp_network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, count_class)
        ).to(device)
        self.network = nn.Linear(input_dim, count_class).to(device)
        self.backbone = BertModel.from_pretrained(backbone).to(device)
        if fix:
            for name, param in list(self.backbone.named_parameters()):
                param.required_grad = False
        self.prompt_module = PromptModule(input_dim, embed=self.backbone.embeddings, device=device)
        if prompt:
            self.backbone.set_input_embeddings = self.prompt_module
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
        self.count_classes = count_class
        self.input_dim = input_dim
        self.fix = fix

    def forward(self, batch):
        tokenized = batch
        # (bsz, L, dim)
        encoded = self.backbone(tokenized)[0][:, 0, :]
        logits = self.network(encoded)
        return logits

    def get_parameter_keys(self):
        """
        获取全部的需要被计算的参数
        """
        if not self.fix:
            return self.state_dict().keys()
        else:
            key_list = self.state_dict().keys()
            result = []
            for key in key_list:
                if key.find("network") != -1 or key.find("prompt") != -1:
                    result.append(key)
            return result
