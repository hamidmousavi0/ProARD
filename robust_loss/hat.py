import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from robust_loss.sat import ctx_noparamgrad_and_eval
def hat_loss(model, x, y, optimizer, step_size=0.007, epsilon=0.031, perturb_steps=10,  beta=1.0, 
             attack='linf',natural_criterion= None ,h=3.5, gamma=1.0, hr_model=None):
    """
    TRADES + Helper-based adversarial training.
    """
  
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
  
    x_adv = x.detach() + 0.001 * torch.randn(x.shape).cuda().detach()
    p_natural = F.softmax(model(x), dim=1)
  
    if attack == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), p_natural)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif attack == 'l2':
        delta = 0.001 * torch.randn(x.shape).cuda().detach()        
        delta = Variable(delta.data, requires_grad=True)
        
        batch_size = len(x)
        optimizer_delta = torch.optim.SGD([delta], lr=step_size)

        for _ in range(perturb_steps):
            adv = x + delta
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1), p_natural)
            loss.backward(retain_graph=True)
          
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            delta.data.add_(x)
            delta.data.clamp_(0, 1).sub_(x)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x + delta, requires_grad=False)
    else:
        raise ValueError(f'Attack={attack} not supported for TRADES training!')
    model.train()

  
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    x_hr = x + h * (x_adv - x)
    if hr_model == None:
        with ctx_noparamgrad_and_eval(model):
            y_hr = model(x_adv).argmax(dim=1)
    else:        
        with ctx_noparamgrad_and_eval(hr_model):
            y_hr = hr_model(x_adv).argmax(dim=1)
  
    optimizer.zero_grad()
  
    out_clean, out_adv, out_help = model(x), model(x_adv), model(x_hr)
    loss_clean = F.cross_entropy(out_clean, y, reduction='mean')
    loss_adv = (1/len(x)) * criterion_kl(F.log_softmax(out_adv, dim=1), F.softmax(out_clean, dim=1))
  
    loss_help = F.cross_entropy(out_help, y_hr, reduction='mean')
    loss = loss_clean + beta * loss_adv + gamma * loss_help
     
    
  
    return loss