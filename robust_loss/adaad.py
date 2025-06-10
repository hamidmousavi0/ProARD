import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
def adaad_inner_loss(model,
                     teacher_model,
                     x_natural,
                     step_size=2/255,
                     steps=10,
                     epsilon=8/255,
                     BN_eval=True,
                     random_init=True,
                     clip_min=0.0,
                     clip_max=1.0):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='none')
    if BN_eval:
        model.eval()

    # set eval mode for teacher model
    teacher_model.eval()
    # generate adversarial example
    if random_init:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    else:
        x_adv = x_natural.detach()
    for _ in range(steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(teacher_model(x_adv), dim=1))
            loss_kl = torch.sum(loss_kl)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural -
                          epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    if BN_eval:
        model.train()
    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max),
                     requires_grad=False)
    return x_adv
def adaad_loss(teacher_model,model,x_natural,y,optimizer,step_size=0.0078,
                epsilon=0.031,
                perturb_steps=10,
                beta = 6.0,
                AdaAD_alpha=1.0):
    adv_inputs  = adaad_inner_loss(model,teacher_model,x_natural,step_size,perturb_steps,epsilon)
    ori_outputs = model(x_natural)
    adv_outputs = model(adv_inputs)
    with torch.no_grad():
        teacher_model.eval()
        t_ori_outputs = teacher_model(x_natural)
        t_adv_outputs = teacher_model(adv_inputs)
    kl_loss1 = nn.KLDivLoss()(F.log_softmax(adv_outputs, dim=1),
                                          F.softmax(t_adv_outputs.detach(), dim=1))
    kl_loss2 = nn.KLDivLoss()(F.log_softmax(ori_outputs, dim=1),
                                          F.softmax(t_ori_outputs.detach(), dim=1))
    loss = AdaAD_alpha*kl_loss1 + (1-AdaAD_alpha)*kl_loss2
    return loss