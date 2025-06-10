import torch
import torch.nn as nn
from typing import Dict
from utils.distributed import DistributedMetric
from tqdm import tqdm
from torchpack import distributed as dist
from utils import accuracy
import copy
import torch.nn.functional as F
import numpy as np
def eval_local_lip(model, x, xp, top_norm=1, btm_norm=float('inf'), reduction='mean'):
    model.eval()
    down = torch.flatten(x - xp, start_dim=1)
    with torch.no_grad():
        if top_norm == "kl":
            criterion_kl = nn.KLDivLoss(reduction='none')
            top = criterion_kl(F.log_softmax(model(xp), dim=1),
                               F.softmax(model(x), dim=1))
            ret = torch.sum(top, dim=1) / torch.norm(down + 1e-6, dim=1, p=btm_norm)
        else:
            top = torch.flatten(model(x), start_dim=1) - torch.flatten(model(xp), start_dim=1)
            ret = torch.norm(top, dim=1, p=top_norm) / torch.norm(down + 1e-6, dim=1, p=btm_norm)

    if reduction == 'mean':
        return torch.mean(ret)
    elif reduction == 'sum':
        return torch.sum(ret)
    else:
        raise ValueError("Not supported reduction")  