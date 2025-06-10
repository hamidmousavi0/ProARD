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
from proard.classification.elastic_nn.networks import DYNMobileNetV3,DYNProxylessNASNets,DYNResNets,DYNProxylessNASNets_Cifar,DYNMobileNetV3_Cifar,DYNResNets_Cifar
from proard.classification.run_manager import DistributedClassificationRunConfig
from proard.classification.run_manager.distributed_run_manager import (
    DistributedRunManager
)
from proard.utils import download_url, MyRandomResizedCrop
from proard.classification.elastic_nn.training.progressive_shrinking import load_models

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    default="expand",
    choices=[
        "kernel", # for architecture except ResNet
        "depth",
        "expand",
        "width", # only for ResNet
    ],
)
parser.add_argument("--phase", type=int, default=2, choices=[1, 2])
parser.add_argument("--resume", action="store_true")
parser.add_argument("--model_name", type=str, default="MBV2", choices=["ResNet50", "MBV3", "ProxylessNASNet","MBV2"])
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
if args.model_name == "ResNet50":
    args.ks_list = "3"
    if args.task == "width":
        if args.robust_mode:
            args.path = "exp/robust/"+ args.dataset + '/' + args.model_name +'/' + args.train_criterion +"/normal2width"
        else:
            args.path = "exp/"+ args.dataset + '/' +args.model_name +'/' + args.train_criterion +"/normal2width"    
        args.dynamic_batch_size = 1
        args.n_epochs = 120
        args.base_lr = 3e-2
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.width_mult_list = "0.65,0.8,1.0"
        args.expand_list = "0.35"
        args.depth_list = "2"
    elif args.task == "depth":
        if args.robust_mode:
            args.path = "exp/robust/"+ args.dataset + '/'  + args.model_name +'/' + args.train_criterion +"/width2width_depth/phase%d" % args.phase
        else:
            args.path = "exp/"+ args.dataset + '/'  + args.model_name +'/' + args.train_criterion +"/width2width_depth/phase%d" % args.phase     
        args.dynamic_batch_size = 2
        if args.phase == 1:
            args.n_epochs = 25
            args.base_lr = 2.5e-3
            args.warmup_epochs = 0
            args.warmup_lr = -1
            args.width_mult_list = "0.65,0.8,1.0"
            args.expand_list ="0.35"
            args.depth_list = "1,2"
        else:
            args.n_epochs = 120
            args.base_lr = 7.5e-3
            args.warmup_epochs = 5
            args.warmup_lr = -1
            args.width_mult_list = "0.65,0.8,1.0"
            args.expand_list = "0.35"
            args.depth_list = "0,1,2"
    elif args.task == "expand":
        if args.robust_mode : 
            args.path = "exp/robust/"+ args.dataset + '/' + args.model_name +'/' + args.train_criterion +"/width_depth2width_depth_width/phase%d" % args.phase 
        else:
            args.path = "exp/"+ args.dataset + '/' + args.model_name +'/' + args.train_criterion +"/width_depth2width_depth_width/phase%d" % args.phase    
        args.dynamic_batch_size = 4
        if args.phase == 1:
            args.n_epochs = 25
            args.base_lr = 2.5e-3
            args.warmup_epochs = 0
            args.warmup_lr = -1
            args.width_mult_list = "0.65,0.8,1.0"
            args.expand_list = "0.25,0.35"
            args.depth_list = "0,1,2"
        else:
            args.n_epochs = 120
            args.base_lr = 7.5e-3
            args.warmup_epochs = 5
            args.warmup_lr = -1
            args.width_mult_list = "0.65,0.8,1.0"
            args.expand_list = "0.2,0.25,0.35"
            args.depth_list = "0,1,2"
    else:
        raise NotImplementedError
else:
    args.width_mult_list = "1.0"    
    if args.task == "kernel":
        if args.robust_mode:
            args.path = "exp/robust/"+ args.dataset + '/' +  args.model_name +'/' + args.train_criterion +"/normal2kernel"
        else:    
            args.path = "exp/"+ args.dataset + '/' +  args.model_name +'/' + args.train_criterion +"/normal2kernel"
        args.dynamic_batch_size = 1
        args.n_epochs = 120
        args.base_lr = 3e-2
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.ks_list = "3,5,7"
        args.expand_list = "6"
        args.depth_list = "4"
    elif args.task == "depth":
        if args.robust_mode : 
            args.path = "exp/robust/"+args.dataset + '/' + args.model_name +'/' + args.train_criterion +"/kernel2kernel_depth/phase%d" % args.phase
        else:
            args.path = "exp/"+args.dataset + '/' + args.model_name +'/' + args.train_criterion +"/kernel2kernel_depth/phase%d" % args.phase    
        args.dynamic_batch_size = 2
        if args.phase == 1:
            args.n_epochs = 25
            args.base_lr = 2.5e-3
            args.warmup_epochs = 0
            args.warmup_lr = -1
            args.ks_list = "3,5,7"
            args.expand_list = "6"
            args.depth_list = "3,4"
        else:
            args.n_epochs = 120
            args.base_lr = 7.5e-3
            args.warmup_epochs = 5
            args.warmup_lr = -1
            args.ks_list = "3,5,7"
            args.expand_list = "6"
            args.depth_list = "2,3,4"
    elif args.task == "expand":
        if args.robust_mode:
            args.path = "exp/robust/"+ args.dataset + '/' + args.model_name +'/' + args.train_criterion +"/kernel_depth2kernel_depth_width/phase%d" % args.phase
        else:
            args.path = "exp/"+ args.dataset + '/' + args.model_name +  '/' + args.train_criterion + "/kernel_depth2kernel_depth_width/phase%d" % args.phase    
        args.dynamic_batch_size = 4
        if args.phase == 1:
            args.n_epochs = 25
            args.base_lr = 2.5e-3
            args.warmup_epochs = 0
            args.warmup_lr = -1
            args.ks_list = "3,5,7"
            args.expand_list = "4,6"
            args.depth_list = "2,3,4"
        else:
            args.n_epochs = 120
            args.base_lr = 7.5e-3
            args.warmup_epochs = 5
            args.warmup_lr = -1
            args.ks_list = "3,5,7"
            args.expand_list = "3,4,6"
            args.depth_list = "2,3,4"
    else:
        raise NotImplementedError
args.manual_seed = 0

args.lr_schedule_type = "cosine"

args.base_batch_size = 64
args.valid_size = 64

args.opt_type = "sgd"
args.momentum = 0.9
args.no_nesterov = False
args.weight_decay = 3e-5
args.label_smoothing = 0.1
args.no_decay_keys = "bn#bias"
args.fp16_allreduce = False

args.model_init = "he_fout"
args.validation_frequency = 1
args.print_frequency = 10

args.n_worker = 8
args.resize_scale = 0.08
args.distort_color = "tf"
if args.dataset == "imagenet":  
    args.image_size = "128,160,192,224"
else:
    args.image_size = "32"    
args.continuous_size = True
args.not_sync_distributed_image_size = False

args.bn_momentum = 0.1
args.bn_eps = 1e-5
args.dropout = 0.1
args.base_stage_width = "google"


args.dy_conv_scaling_mode = 1
args.independent_distributed_sampling = False

args.kd_ratio = 1.0
args.kd_type = "ce"


if __name__ == "__main__":
    os.makedirs(args.path, exist_ok=True)

    # Initialize Horovod
    hvd.init()
    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())
    if args.robust_mode:
        args.teacher_path = 'exp/robust/teacher/' + args.dataset + '/' +  args.model_name + '/' + args.train_criterion + "/checkpoint/model_best.pth.tar"
    else:
        args.teacher_path = 'exp/teacher/' + args.dataset + '/' +  args.model_name +'/' + args.train_criterion + "/checkpoint/model_best.pth.tar"
    num_gpus = hvd.size()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    # image size
    args.image_size = [int(img_size) for img_size in args.image_size.split(",")]
    if len(args.image_size) == 1:
        args.image_size = args.image_size[0]
    MyRandomResizedCrop.CONTINUOUS = args.continuous_size
    MyRandomResizedCrop.SYNC_DISTRIBUTED = not args.not_sync_distributed_image_size

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
    args.test_batch_size = args.base_batch_size * 4
    run_config = DistributedClassificationRunConfig(
        **args.__dict__, num_replicas=num_gpus, rank=hvd.rank()
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
    if args.model_name == "ResNet50":
        args.expand_list = [float(e) for e in args.expand_list.split(",")]
    else:
        args.expand_list = [int(e) for e in args.expand_list.split(",")]    
    args.depth_list = [int(d) for d in args.depth_list.split(",")]

    args.width_mult_list = (
        args.width_mult_list[0]
        if len(args.width_mult_list) == 1
        else args.width_mult_list
    )

    if args.model_name == "ResNet50":
        if args.dataset == "cifar10" or args.dataset == "cifar100":
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
                                dropout_rate= args.dropout, ks_list=args.ks_list , expand_ratio_list= args.expand_list , depth_list= args.depth_list,width_mult=args.width_mult_list)
        else:
            net = DYNMobileNetV3(n_classes=run_config.data_provider.n_classes,bn_param=(args.bn_momentum,args.bn_eps),
                                dropout_rate= args.dropout, ks_list=args.ks_list , expand_ratio_list= args.expand_list , depth_list= args.depth_list,width_mult=args.width_mult_list)    
    elif args.model_name == "ProxylessNASNet": 
        if args.dataset == "cifar10" or args.dataset == "cifar100":     
            net = DYNProxylessNASNets_Cifar(n_classes=run_config.data_provider.n_classes,bn_param=(args.bn_momentum,args.bn_eps),
                                dropout_rate= args.dropout, ks_list=args.ks_list , expand_ratio_list= args.expand_list , depth_list= args.depth_list,width_mult=args.width_mult_list)
        else:
            net = DYNProxylessNASNets(n_classes=run_config.data_provider.n_classes,bn_param=(args.bn_momentum,args.bn_eps),
                                dropout_rate= args.dropout, ks_list=args.ks_list , expand_ratio_list= args.expand_list , depth_list= args.depth_list,width_mult=args.width_mult_list)   
    elif args.model_name == "MBV2": 
        if args.dataset == "cifar10" or args.dataset == "cifar100":     
            net = DYNProxylessNASNets_Cifar(n_classes=run_config.data_provider.n_classes,bn_param=(args.bn_momentum,args.bn_eps),
                                dropout_rate= args.dropout, ks_list=args.ks_list , expand_ratio_list= args.expand_list , depth_list= args.depth_list,width_mult=args.width_mult_list,base_stage_width=args.base_stage_width)
        else:
            net = DYNProxylessNASNets(n_classes=run_config.data_provider.n_classes,bn_param=(args.bn_momentum,args.bn_eps),
                                dropout_rate= args.dropout, ks_list=args.ks_list , expand_ratio_list= args.expand_list , depth_list= args.depth_list,width_mult=args.width_mult_list,base_stage_width=args.base_stage_width)             
    else: 
        raise NotImplementedError  
    # teacher model
    if args.kd_ratio > 0:
        
        if args.model_name =="ResNet50":
            if args.dataset == "cifar10" or args.dataset == "cifar100":
                args.teacher_model = DYNResNets_Cifar(
                    n_classes=run_config.data_provider.n_classes,
                    bn_param=(args.bn_momentum, args.bn_eps),
                    dropout_rate=args.dropout,
                    depth_list=[2],
                    expand_ratio_list=[0.35],
                    width_mult_list=[1.0],
                )
            else:
                args.teacher_model = DYNResNets(
                    n_classes=run_config.data_provider.n_classes,
                    bn_param=(args.bn_momentum, args.bn_eps),
                    dropout_rate=args.dropout,
                    depth_list=[2],
                    expand_ratio_list=[0.35],
                    width_mult_list=[1.0],
                )   
        elif args.model_name =="MBV3":    
            if args.dataset == "cifar10" or args.dataset == "cifar100":
                args.teacher_model = DYNMobileNetV3_Cifar(
                    n_classes=run_config.data_provider.n_classes,
                    bn_param=(args.bn_momentum, args.bn_eps),
                    dropout_rate=0,
                    width_mult=1.0,
                    ks_list=[7],
                    expand_ratio_list=[6],
                    depth_list=[4]
                )
            else:
                args.teacher_model = DYNMobileNetV3(
                    n_classes=run_config.data_provider.n_classes,
                    bn_param=(args.bn_momentum, args.bn_eps),
                    dropout_rate=0,
                    width_mult=1.0,
                    ks_list=[7],
                    expand_ratio_list=[6],
                    depth_list=[4]
                )    
        elif args.model_name == "ProxylessNASNet":
            if args.dataset == "cifar10" or args.dataset == "cifar100":
                args.teacher_model  = DYNProxylessNASNets_Cifar(n_classes=run_config.data_provider.n_classes,
                    bn_param=(args.bn_momentum, args.bn_eps),
                    dropout_rate=0,
                    width_mult=1.0,
                    ks_list=[7],
                    expand_ratio_list=[6],
                    depth_list=[4])   
            else:
                args.teacher_model  = DYNProxylessNASNets(n_classes=run_config.data_provider.n_classes,
                    bn_param=(args.bn_momentum, args.bn_eps),
                    dropout_rate=0,
                    width_mult=1.0,
                    ks_list=[7],
                    expand_ratio_list=[6],
                    depth_list=[4]) 
        elif args.model_name == "MBV2":
            if args.dataset == "cifar10" or args.dataset == "cifar100":
                args.teacher_model  = DYNProxylessNASNets_Cifar(n_classes=run_config.data_provider.n_classes,
                    bn_param=(args.bn_momentum, args.bn_eps),
                    dropout_rate=0,
                    width_mult=1.0,
                    ks_list=[7],
                    expand_ratio_list=[6],
                    depth_list=[4],base_stage_width=args.base_stage_width)   
            else:
                args.teacher_model  = DYNProxylessNASNets(n_classes=run_config.data_provider.n_classes,
                    bn_param=(args.bn_momentum, args.bn_eps),
                    dropout_rate=0,
                    width_mult=1.0,
                    ks_list=[7],
                    expand_ratio_list=[6],
                    depth_list=[4],base_stage_width=args.base_stage_width)                 
        args.teacher_model.cuda()
    
    """ Distributed RunManager """
    # Horovod: (optional) compression algorithm.
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
    # hvd broadcast
    distributed_run_manager.broadcast()

    # load teacher net weights
    if args.kd_ratio > 0:
        load_models(
            distributed_run_manager, args.teacher_model, model_path=args.teacher_path
        )

    # training
    from proard.classification.elastic_nn.training.progressive_shrinking import (
        validate,
        train,
    )
    if args.model_name =="ResNet50":
        validate_func_dict = {
            "image_size_list": {224 if args.dataset == "imagenet" else 32}
            if isinstance(args.image_size, int)
            else sorted({160, 224}),
            "width_mult_list": sorted({min(args.width_mult_list), max(args.width_mult_list)}),
            "expand_ratio_list": sorted({min(args.expand_list), max(args.expand_list)}),
            "depth_list": sorted({min(net.depth_list), max(net.depth_list)}),
        }
    else:
        validate_func_dict = {
            "image_size_list": {224 if args.dataset == "imagenet" else 32}
            if isinstance(args.image_size, int)
            else sorted({160, 224}),
            "width_mult_list": [1.0],
            "ks_list": sorted({min(args.ks_list), max(args.ks_list)}),
            "expand_ratio_list": sorted({min(args.expand_list), max(args.expand_list)}),
            "depth_list": sorted({min(net.depth_list), max(net.depth_list)}),
        }  

    if args.task == "width":
        from proard.classification.elastic_nn.training.progressive_shrinking import (
            train_elastic_width_mult,
        )
        if distributed_run_manager.start_epoch == 0:
            if args.robust_mode:
                args.dyn_checkpoint_path ='exp/robust/teacher/' +args.dataset + '/' +  args.model_name +'/' + args.train_criterion + "/checkpoint/model_best.pth.tar"
            else:
                args.dyn_checkpoint_path ='exp/teacher/' +args.dataset + '/' +  args.model_name +'/' + args.train_criterion + "/checkpoint/model_best.pth.tar"   
            load_models(
                distributed_run_manager,
                distributed_run_manager.net,
                args.dyn_checkpoint_path,
            )
            distributed_run_manager.write_log(
                "%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s"
                % validate(distributed_run_manager, is_test=True, **validate_func_dict),
                "valid",
            )
        else:
            assert args.resume
        train_elastic_width_mult (train,distributed_run_manager,args,validate_func_dict)    



    elif args.task == "kernel":
        validate_func_dict["ks_list"] = sorted(args.ks_list)
        if distributed_run_manager.start_epoch == 0:
            if args.robust_mode:
                args.dyn_checkpoint_path ='exp/robust/teacher/' + args.dataset + '/' +  args.model_name +'/' + args.train_criterion + "/checkpoint/model_best.pth.tar"
            else: 
                args.dyn_checkpoint_path ='exp/teacher/' + args.dataset + '/' +  args.model_name +'/' + args.train_criterion + "/checkpoint/model_best.pth.tar"    
            load_models(
                distributed_run_manager,
                distributed_run_manager.net,
                args.dyn_checkpoint_path,
            )
            distributed_run_manager.write_log(
               "%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s"
                % validate(distributed_run_manager, is_test=True, **validate_func_dict),
                "valid",
            )
        else:
            assert args.resume
        train(
            distributed_run_manager,
            args,
            lambda _run_manager, epoch, is_test: validate(
                _run_manager, epoch, is_test, **validate_func_dict
            ),
        )
    elif args.task == "depth":
        from proard.classification.elastic_nn.training.progressive_shrinking import (
            train_elastic_depth,
        )
        if args.robust_mode:
            if args.model_name =="ResNet50":
                if args.phase == 1:
                    args.dyn_checkpoint_path =  "exp/robust/"+ args.dataset + '/'+ args.model_name +'/' + args.train_criterion +"/normal2width" +"/checkpoint/model_best.pth.tar"
                else:
                    args.dyn_checkpoint_path = "exp/robust/"+ args.dataset + '/' + args.model_name +'/' + args.train_criterion +"/width2width_depth/phase1" + "/checkpoint/model_best.pth.tar"
            else:
                if args.phase == 1:
                    args.dyn_checkpoint_path =  "exp/robust/"+ args.dataset + '/' +  args.model_name +'/' + args.train_criterion +"/normal2kernel" +"/checkpoint/model_best.pth.tar"
                else:
                    args.dyn_checkpoint_path = "exp/robust/"+ args.dataset + '/' + args.model_name +'/' + args.train_criterion +"/kernel2kernel_depth/phase1" + "/checkpoint/model_best.pth.tar"     
        else :
            if args.model_name =="ResNet50":
                if args.phase == 1:
                    args.dyn_checkpoint_path =  "exp/"+ args.dataset + '/'+ args.model_name +'/' + args.train_criterion +"/normal2width" +"/checkpoint/model_best.pth.tar"
                else:
                    args.dyn_checkpoint_path = "exp/"+ args.dataset + '/' + args.model_name +'/' + args.train_criterion +"/width2width_depth/phase1" + "/checkpoint/model_best.pth.tar"
            else:
                if args.phase == 1:
                    args.dyn_checkpoint_path =  "exp/"+ args.dataset + '/' +  args.model_name +'/' + args.train_criterion +"/normal2kernel" +"/checkpoint/model_best.pth.tar"
                else:
                    args.dyn_checkpoint_path = "exp/"+ args.dataset + '/' + args.model_name +'/' + args.train_criterion +"/kernel2kernel_depth/phase1" + "/checkpoint/model_best.pth.tar"                  
        train_elastic_depth(train, distributed_run_manager, args, validate_func_dict)
    elif args.task == "expand":
        from proard.classification.elastic_nn.training.progressive_shrinking import (
            train_elastic_expand,
        )
        if args.robust_mode : 
            if args.model_name =="ResNet50":
                if args.phase == 1:
                    args.dyn_checkpoint_path =  "exp/robust/"+ args.dataset + '/'+ args.model_name +'/' + args.train_criterion +"/width2width_depth/phase2" + "/checkpoint/model_best.pth.tar"
                else:
                    args.dyn_checkpoint_path = "exp/robust/"+ args.dataset + '/'+ args.model_name +'/' + args.train_criterion +"/width_depth2width_depth_width/phase1" + "/checkpoint/model_best.pth.tar"
            else:
                if args.phase == 1:
                    args.dyn_checkpoint_path =  "exp/robust/"+ args.dataset + '/'+ args.model_name +'/' + args.train_criterion +"/kernel2kernel_depth/phase2" + "/checkpoint/model_best.pth.tar"  
                else:
                    args.dyn_checkpoint_path = "exp/robust/"+ args.dataset + '/'+ args.model_name +'/' + args.train_criterion +"/kernel_depth2kernel_depth_width/phase1" +  "/checkpoint/model_best.pth.tar" 
        else:
            if args.model_name =="ResNet50":
                if args.phase == 1:
                    args.dyn_checkpoint_path =  "exp/"+ args.dataset + '/'+ args.model_name +'/' + args.train_criterion +"/width2width_depth/phase2" + "/checkpoint/model_best.pth.tar"
                else:
                    args.dyn_checkpoint_path = "exp/"+ args.dataset + '/'+ args.model_name +'/' + args.train_criterion +"/width_depth2width_depth_width/phase1" + "/checkpoint/model_best.pth.tar"
            else:
                if args.phase == 1:
                    args.dyn_checkpoint_path =  "exp/"+ args.dataset + '/'+ args.model_name +'/' + args.train_criterion +"/kernel2kernel_depth/phase2" + "/checkpoint/model_best.pth.tar"  
                else:
                    args.dyn_checkpoint_path = "exp/"+ args.dataset + '/'+ args.model_name +'/' + args.train_criterion +"/kernel_depth2kernel_depth_width/phase1" +  "/checkpoint/model_best.pth.tar" 

        train_elastic_expand(train, distributed_run_manager, args, validate_func_dict)
    else:
        raise NotImplementedError
