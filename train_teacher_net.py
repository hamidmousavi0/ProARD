# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import argparse
import numpy as np
import os
import random
# using for distributed training
import horovod.torch as hvd
import torch

from proard.classification.elastic_nn.modules.dynamic_op import (
    DynamicSeparableConv2d,
)
from proard.classification.elastic_nn.networks import DYNResNets,DYNMobileNetV3,DYNProxylessNASNets,DYNMobileNetV3_Cifar,DYNResNets_Cifar,DYNProxylessNASNets_Cifar
from proard.classification.run_manager import DistributedClassificationRunConfig
from proard.classification.networks import WideResNet
from proard.classification.run_manager import DistributedRunManager


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="MBV2", choices=["ResNet50", "MBV3", "ProxylessNASNet","WideResNet","MBV2"])
parser.add_argument("--teacher_model_name", type=str, default="WideResNet", choices=["WideResNet"])
parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100", "imagenet"])
parser.add_argument("--robust_mode", type=bool, default=True)
parser.add_argument("--epsilon", type=float, default=0.031)
parser.add_argument("--num_steps", type=int, default=10)
parser.add_argument("--step_size", type=float, default=0.0078) 
parser.add_argument("--clip_min", type=int, default=0)
parser.add_argument("--clip_max", type=int, default=1)
parser.add_argument("--const_init", type=bool, default=False)
parser.add_argument("--beta", type=float, default=6.0)
parser.add_argument("--distance", type=str, default="l_inf",choices=["l_inf","l2"])
parser.add_argument("--train_criterion", type=str, default="trades",choices=["trades","sat","mart","hat"])
parser.add_argument("--test_criterion", type=str, default="ce",choices=["ce"])
parser.add_argument("--kd_criterion", type=str, default="rslad",choices=["ard","rslad","adaad"])
parser.add_argument("--attack_type", type=str, default="linf-pgd",choices=['fgsm', 'linf-pgd', 'fgm', 'l2-pgd', 'linf-df', 'l2-df', 'linf-apgd', 'l2-apgd','squar_attack','autoattack','apgd_ce'])


args = parser.parse_args()
if args.robust_mode:    
    args.path = 'exp/robust/teacher/' + args.dataset + "/" +  args.model_name + '/' + args.train_criterion
else:
    args.path = 'exp/teacher/' + args.dataset + "/" +  args.model_name
args.n_epochs = 120
args.base_lr = 0.1
args.warmup_epochs = 5
args.warmup_lr = -1
args.manual_seed = 0
args.lr_schedule_type = "cosine"
args.base_batch_size = 128
args.valid_size = None
args.opt_type = "sgd"
args.momentum = 0.9
args.no_nesterov = False
args.weight_decay = 2e-4
args.label_smoothing = 0.0
args.no_decay_keys = "bn#bias"
args.fp16_allreduce = False
args.model_init = "he_fout"
args.validation_frequency = 1
args.print_frequency = 10
args.n_worker = 32
if args.dataset =="imagenet":
    args.image_size = "224"
else:
    args.image_size = "32"     
args.continuous_size = True
args.not_sync_distributed_image_size = False
args.bn_momentum = 0.1
args.bn_eps = 1e-5
args.dropout = 0.0
args.base_stage_width = "google"
###### Parameters for MBV3, ProxylessNet, and MBV2
if args.model_name != "ResNet50":
    args.ks_list = '7'
    args.expand_list = '6'
    args.depth_list = '4'
    args.width_mult_list = "1.0"
else: 
    ###### Parameters for ResNet50
    args.ks_list = "3"
    args.expand_list = "0.35"
    args.depth_list = "2"
    args.width_mult_list = "1.0"
########################################
args.dy_conv_scaling_mode = 1
args.independent_distributed_sampling = False
args.kd_ratio = 0.0
args.kd_type = "ce"
args.dynamic_batch_size = 1
args.num_gpus = 4
if __name__ == "__main__":
    os.makedirs(args.path, exist_ok=True)

    # Initialize Horovod
    hvd.init()
    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())

    num_gpus = hvd.size()
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    # image size
    args.image_size = [int(img_size) for img_size in args.image_size.split(",")]
    if len(args.image_size) == 1:
        args.image_size = args.image_size[0]

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        "momentum": args.momentum,
        "nesterov": not args.no_nesterov,
    }
    args.init_lr = args.base_lr * num_gpus  # linearly rescale the learning rate
    if args.warmup_lr < 0:
        args.warmup_lr = args.base_lr
    args.train_batch_size = args.base_batch_size
    args.test_batch_size = args.base_batch_size 
    print(args.__dict__)
    run_config = DistributedClassificationRunConfig(
        **args.__dict__,num_replicas=num_gpus, rank=hvd.rank()
    )

    # print run config information
    if hvd.rank() == 0:
        print("Run config:")
        for k, v in run_config.config.items():
            print("\t%s: %s" % (k, v))

    if args.dy_conv_scaling_mode == -1:
        args.dy_conv_scaling_mode = None
    DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = args.dy_conv_scaling_mode

    # build net from args
    args.width_mult_list = [
        float(width_mult) for width_mult in args.width_mult_list.split(",")
    ]
    args.ks_list = [int(ks) for ks in args.ks_list.split(",")]
    args.expand_list = [float(e) for e in args.expand_list.split(",")]
    args.depth_list = [int(d) for d in args.depth_list.split(",")]

    args.width_mult_list = (
        args.width_mult_list[0]
        if len(args.width_mult_list) == 1
        else args.width_mult_list
    )
    if args.model_name == "ResNet50":
        if args.dataset == "cifar10" or args.dataset == "cifar100":
            # net = ResNet50_Cifar(n_classes=run_config.data_provider.n_classes)
            net = DYNResNets_Cifar( n_classes=run_config.data_provider.n_classes,
                bn_param=(args.bn_momentum, args.bn_eps),
                dropout_rate=args.dropout,
                depth_list=args.depth_list,
                expand_ratio_list=args.expand_list,
                width_mult_list=args.width_mult_list,)
        else:
            net = DYNResNets( n_classes=run_config.data_provider.n_classes,
                bn_param=(args.bn_momentum, args.bn_eps),
                dropout_rate=args.dropout,
                depth_list=args.depth_list,
                expand_ratio_list=args.expand_list,
                width_mult_list=args.width_mult_list,)   
    elif args.model_name == "MBV3":  
        if args.dataset == "cifar10" or args.dataset == "cifar100":  
            net = DYNMobileNetV3_Cifar(n_classes=run_config.data_provider.n_classes,bn_param=(args.bn_momentum,args.bn_eps),
                                dropout_rate= args.dropout, ks_list=args.ks_list , expand_ratio_list= args.expand_list , depth_list= args.depth_list)
        else:
            net = DYNMobileNetV3(n_classes=run_config.data_provider.n_classes,bn_param=(args.bn_momentum,args.bn_eps),
                                dropout_rate= args.dropout, ks_list=args.ks_list , expand_ratio_list= args.expand_list , depth_list= args.depth_list)    
    elif args.model_name == "ProxylessNASNet": 
        if args.dataset == "cifar10" or args.dataset == "cifar100":     
            net = DYNProxylessNASNets_Cifar(n_classes=run_config.data_provider.n_classes,bn_param=(args.bn_momentum,args.bn_eps),
                                dropout_rate= args.dropout, ks_list=args.ks_list , expand_ratio_list= args.expand_list , depth_list= args.depth_list)
        else:
            net = DYNProxylessNASNets(n_classes=run_config.data_provider.n_classes,bn_param=(args.bn_momentum,args.bn_eps),
                                dropout_rate= args.dropout, ks_list=args.ks_list , expand_ratio_list= args.expand_list , depth_list= args.depth_list) 
             
    elif args.model_name == "MBV2": 
        if args.dataset == "cifar10" or args.dataset == "cifar100":     
            net = DYNProxylessNASNets_Cifar(n_classes=run_config.data_provider.n_classes,bn_param=(args.bn_momentum,args.bn_eps),base_stage_width=args.base_stage_width,
                                dropout_rate= args.dropout, ks_list=args.ks_list , expand_ratio_list= args.expand_list , depth_list= args.depth_list)
        else:
            net = DYNProxylessNASNets(n_classes=run_config.data_provider.n_classes,bn_param=(args.bn_momentum,args.bn_eps),base_stage_width=args.base_stage_width,
                                dropout_rate= args.dropout, ks_list=args.ks_list , expand_ratio_list= args.expand_list , depth_list= args.depth_list) 
    else: 
        raise NotImplementedError                    
    if args.teacher_model_name == "WideResNet":  
        if args.dataset == "cifar10" or args.dataset == "cifar100":  
            net = WideResNet(num_classes=run_config.data_provider.n_classes)
        else:
           raise NotImplementedError 
    else: 
        raise NotImplementedError  
    args.teacher_model = None #'exp/teacher/' + args.dataset + "/" +  "WideResNet"
    
    """ Distributed RunManager """
    #Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    distributed_run_manager = DistributedRunManager(
        args.path,
        net,
        run_config,
        compression,
        backward_steps=args.dynamic_batch_size,
        is_root=(hvd.rank() == 0),
    )
    distributed_run_manager.save_config()
    distributed_run_manager.broadcast()

   
    distributed_run_manager.train(args)
