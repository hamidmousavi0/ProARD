import torch
import torch.nn as nn
import torch.nn.functional as F
from attacks import create_attack
import numpy as np
from torch.autograd import Variable
from contextlib import contextmanager
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class ctx_noparamgrad(object):
    def __init__(self, module):
        self.prev_grad_state = get_param_grad_state(module)
        self.module = module
        set_param_grad_off(module)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        set_param_grad_state(self.module, self.prev_grad_state)
        return False


class ctx_eval(object):
    def __init__(self, module):
        self.prev_training_state = get_module_training_state(module)
        self.module = module
        set_module_training_off(module)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        set_module_training_state(self.module, self.prev_training_state)
        return False


@contextmanager
def ctx_noparamgrad_and_eval(module):
    with ctx_noparamgrad(module) as a, ctx_eval(module) as b:
        yield (a, b)


def get_module_training_state(module):
    return {mod: mod.training for mod in module.modules()}


def set_module_training_state(module, training_state):
    for mod in module.modules():
        mod.training = training_state[mod]


def set_module_training_off(module):
    for mod in module.modules():
        mod.training = False


def get_param_grad_state(module):
    return {param: param.requires_grad for param in module.parameters()}


def set_param_grad_state(module, grad_state):
    for param in module.parameters():
        param.requires_grad = grad_state[param]


def set_param_grad_off(module):
    for param in module.parameters():
        param.requires_grad = False
class MadrysLoss(nn.Module):
    def __init__(self, step_size=0.007, epsilon=0.031, perturb_steps=10, beta=6.0,
                 distance='l_inf', cutmix=False, adjust_freeze=True, cutout=False,
                 cutout_length=16):
        super(MadrysLoss, self).__init__()
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.beta = beta
        self.distance = distance
        self.cross_entropy =  torch.nn.CrossEntropyLoss()
        self.adjust_freeze = adjust_freeze
        self.cutout = cutout
        self.cutout_length = cutout_length

    def forward(self, model, x_natural, labels): #optimizer
        model.eval()
        if self.adjust_freeze:
            for param in model.parameters():
                param.requires_grad = False

        # generate adversarial example
        x_adv = x_natural.detach() + self.step_size * torch.randn(x_natural.shape).to(device).detach()
        if self.distance == 'l_inf':
            adv_loss = 0
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                loss_ce = self.cross_entropy(model(x_adv), labels)
                grad = torch.autograd.grad(loss_ce, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        x_adv = Variable(x_adv, requires_grad=False)

        if self.adjust_freeze:
            for param in model.parameters():
                param.requires_grad = True

        if self.cutout:
            batch_size = x_adv.shape[0]
            c, h, w = x_adv.shape[1], x_adv.shape[2], x_adv.shape[3]
            mask = torch.ones(batch_size, c, h, w).float()
            for j in range(batch_size):
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - self.cutout_length // 2, 0, h)
                y2 = np.clip(y + self.cutout_length // 2, 0, h)
                x1 = np.clip(x - self.cutout_length // 2, 0, w)
                x2 = np.clip(x + self.cutout_length // 2, 0, w)

                mask[j, :, y1: y2, x1: x2] = 0.0
            x_adv = x_adv * mask.to(device)

        model.train()
        # optimizer.zero_grad()


        logits = model(x_adv)
        loss = self.cross_entropy(logits, labels)

        return loss


def sat_loss(model, x, y,optimizer,step_size,epsilon,num_steps,attack_type,beta,criterion= torch.nn.CrossEntropyLoss()):
    """
    Adversarial training (Madry et al, 2017).
    """
    attack = create_attack(model, criterion, 'linf-pgd', epsilon, num_steps, step_size)  
    with ctx_noparamgrad_and_eval(model):
        x_adv, _ = attack.perturb(x, y)
    print(x_adv.shape)
    y_adv = y
    out = model(x_adv)
    loss = criterion(out, y_adv)
    
    return loss  






