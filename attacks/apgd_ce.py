import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable
from adv_lib.attacks import carlini_wagner_linf
import torch.optim as optim
from autoattack import AutoAttack
import numpy as np
import logging
from .base import Attack,LabelMixin 
from typing import List, Union,Dict

import torch
import torch.nn as nn
from typing import Dict
from .utils import  ctx_noparamgrad_and_eval
from utils.distributed import DistributedMetric
from tqdm import tqdm
from torchpack import distributed as dist
from utils import accuracy


   
class Autoattack_apgd_ce(Attack, LabelMixin):
    
    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            ord=np.inf, targeted=False, rand_init_type='uniform'):
        super(Autoattack_apgd_ce, self).__init__(predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.rand_init_type = rand_init_type
        self.ord = ord
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        self.adversary = AutoAttack(predict, norm='Linf', eps=self.eps, version='standard')         
    def perturb(self, x, y=None):
        self.adversary.attacks_to_run=['apgd-ce']
        adversarial_examples = self.adversary.run_standard_evaluation(x, y, bs=100) 
        return adversarial_examples,adversarial_examples    

    def eval_AutoAttack_apgd_ce(self,data_loader_dict: Dict)-> Dict:

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


