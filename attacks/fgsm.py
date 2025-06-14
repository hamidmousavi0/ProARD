import torch
import torch.nn as nn

from .base import Attack, LabelMixin
from .utils import  ctx_noparamgrad_and_eval
from .utils import batch_multiply
from .utils import clamp ,normalize_by_pnorm
from utils.distributed import DistributedMetric
from tqdm import tqdm
from torchpack import distributed as dist
from utils import accuracy
from typing import Dict
class FGSMAttack(Attack, LabelMixin):
    """
    One step fast gradient sign method (Goodfellow et al, 2014).
    Arguments:
        predict (nn.Module): forward pass function.
        loss_fn (nn.Module): loss function.
        eps (float): attack step size.
        clip_min (float): mininum value per input dimension.
        clip_max (float): maximum value per input dimension.
        targeted (bool): indicate if this is a targeted attack.
    """

    def __init__(self, predict, loss_fn=None, eps=0.3, clip_min=0., clip_max=1., targeted=False):
        super(FGSMAttack, self).__init__(predict, loss_fn, clip_min, clip_max)

        self.eps = eps
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with an attack length of eps.
        Arguments:
            x (torch.Tensor): input tensor.
            y  (torch.Tensor): label tensor.
                - if None and self.targeted=False, compute y as predicted labels.
                - if self.targeted=True, then y must be the targeted labels.
        Returns: 
            torch.Tensor containing perturbed inputs.
            torch.Tensor containing the perturbation.
        """

        x, y = self._verify_and_process_inputs(x, y)
        
        xadv = x.requires_grad_()
        outputs = self.predict(xadv)

        loss = self.loss_fn(outputs, y)
        if self.targeted:
            loss = -loss
        loss.backward()
        grad_sign = xadv.grad.detach().sign()

        xadv = xadv + batch_multiply(self.eps, grad_sign)
        xadv = clamp(xadv, self.clip_min, self.clip_max)
        radv = xadv - x
        return xadv.detach(), radv.detach()


LinfFastGradientAttack = FGSMAttack


class FGMAttack(Attack, LabelMixin):
    """
    One step fast gradient method. Perturbs the input with gradient (not gradient sign) of the loss wrt the input.
    Arguments:
        predict (nn.Module): forward pass function.
        loss_fn (nn.Module): loss function.
        eps (float): attack step size.
        clip_min (float): mininum value per input dimension.
        clip_max (float): maximum value per input dimension.
        targeted (bool): indicate if this is a targeted attack.
    """

    def __init__(self, predict, loss_fn=None, eps=0.3, clip_min=0., clip_max=1., targeted=False):
        super(FGMAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.eps = eps
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with an attack length of eps.
        Arguments:
            x (torch.Tensor): input tensor.
            y  (torch.Tensor): label tensor.
                - if None and self.targeted=False, compute y as predicted labels.
                - if self.targeted=True, then y must be the targeted labels.
        Returns: 
            torch.Tensor containing perturbed inputs.
            torch.Tensor containing the perturbation.
        """

        x, y = self._verify_and_process_inputs(x, y)
        xadv = x.requires_grad_()
        outputs = self.predict(xadv)

        loss = self.loss_fn(outputs, y)
        if self.targeted:
            loss = -loss
        loss.backward()
        grad = normalize_by_pnorm(xadv.grad)
        xadv = xadv + batch_multiply(self.eps, grad)
        xadv = clamp(xadv, self.clip_min, self.clip_max)
        radv = xadv - x

        return xadv.detach(), radv.detach()
    def eval_fgsm(self,data_loader_dict: Dict)-> Dict:

        test_criterion = nn.CrossEntropyLoss().cuda()
        val_loss = DistributedMetric()
        val_top1 = DistributedMetric()
        val_top5 = DistributedMetric()
        val_advloss = DistributedMetric()
        val_advtop1 = DistributedMetric()
        val_advtop5 = DistributedMetric()
        self.predict.eval()
        with tqdm(
                total=len(data_loader_dict["val"]),
                desc="Eval",
                disable=not dist.is_master(),
            ) as t:
                for images, labels in data_loader_dict["val"]:
                    images, labels = images.cuda(), labels.cuda()
                    # compute output
                    output = self.predict(images)
                    loss = test_criterion(output, labels)
                    val_loss.update(loss, images.shape[0])
                    acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                    val_top5.update(acc5[0], images.shape[0])
                    val_top1.update(acc1[0], images.shape[0])
                    with ctx_noparamgrad_and_eval(self.predict):
                        images_adv,_ = self.perturb(images, labels)
                    output_adv = self.predict(images_adv)   
                    loss_adv = test_criterion(output_adv,labels) 
                    val_advloss.update(loss_adv, images.shape[0])   
                    acc1_adv, acc5_adv = accuracy(output_adv, labels, topk=(1, 5))   
                    val_advtop1.update(acc1_adv[0], images.shape[0])
                    val_advtop5.update(acc5_adv[0], images.shape[0])                  
                    t.set_postfix(
                        {
                            "loss": val_loss.avg.item(),
                            "top1": val_top1.avg.item(),
                            "top5": val_top5.avg.item(),
                            "adv_loss": val_advloss.avg.item(),
                            "adv_top1": val_advtop1.avg.item(),
                            "adv_top5": val_advtop5.avg.item(),
                            "#samples": val_top1.count.item(),
                            "batch_size": images.shape[0],
                            "img_size": images.shape[2],
                        }
                    )
                    t.update()

        val_results = {
            "val_top1": val_top1.avg.item(),
            "val_top5": val_top5.avg.item(),
            "val_loss": val_loss.avg.item(),
            "val_advtop1": val_advtop1.avg.item(),
            "val_advtop5": val_advtop5.avg.item(),
            "val_advloss": val_advloss.avg.item(),
        }
        return val_results

L2FastGradientAttack = FGMAttack
