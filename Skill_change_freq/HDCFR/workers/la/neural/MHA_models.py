import torch
from torch import nn
import torch.nn.functional as F


class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

def layer_init_zero(layer):
    nn.init.constant_(layer.weight.data, 0)
    nn.init.constant_(layer.bias.data, 0)
    return layer

def range_tensor(end, config_device):
    return torch.arange(end).long().to(config_device)


############################################# start from here #############################################
class SkillMhaLayer(BaseNet):

    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]  # probably the memory or say embedding matrix will be updated here
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        # "add tgt and then norm" seems to be a module
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class SkillPolicy(BaseNet):

    def __init__(self, dmodel, nhead, nlayers, nhid, dropout):
        super().__init__()
        self.layers = nn.ModuleList([SkillMhaLayer(dmodel, nhead, nhid, dropout) for i in range(nlayers)])
        self.norm = nn.LayerNorm(dmodel)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, memory, tgt):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory)
        output = self.norm(output)
        return output


