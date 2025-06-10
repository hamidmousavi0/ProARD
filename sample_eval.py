# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import torch
import argparse
import sys    
from proard.classification.data_providers.imagenet import ImagenetDataProvider
from proard.classification.data_providers.cifar10 import Cifar10DataProvider
from proard.classification.data_providers.cifar100 import Cifar100DataProvider
from proard.classification.run_manager import ClassificationRunConfig, RunManager,DistributedRunManager
from proard.model_zoo import DYN_net
from proard.nas.accuracy_predictor import AccuracyDataset,AccuracyPredictor,ResNetArchEncoder,RobustnessPredictor,MobileNetArchEncoder,AccuracyRobustnessDataset,Accuracy_Robustness_Predictor

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--path", help="The path of imagenet", type=str, default="/dataset/imagenet"
)
parser.add_argument("-g", "--gpu", help="The gpu(s) to use", type=str, default="all")
parser.add_argument(
    "-b",
    "--batch-size",
    help="The batch on every device for validation",
    type=int,
    default=128,
)
parser.add_argument("-j", "--workers", help="Number of workers", type=int, default=20)
parser.add_argument(
    "-n",
    "--net",
    metavar="DYNNET",
    default="MBV3",
    choices=[
        "ResNet50",
        "MBV3",
        "ProxylessNASNet",
        "MBV2"
    ],
    help="dynamic networks",
)
parser.add_argument(
    "--dataset", type=str, default="cifar10" ,choices=["cifar10", "cifar100", "imagenet"]
)
parser.add_argument("--train_criterion", type=str, default="trades",choices=["trades","sat","mart","hat"])
parser.add_argument(
    "--robust_mode", type=bool, default=True
)
parser.add_argument(
    "--WPS", type=bool, default=False
)
args = parser.parse_args()
if args.gpu == "all":
    device_list = range(torch.cuda.device_count())
    args.gpu = ",".join(str(_) for _ in device_list)
else:
    device_list = [int(_) for _ in args.gpu.split(",")]
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.batch_size = args.batch_size * max(len(device_list), 1)
ImagenetDataProvider.DEFAULT_PATH = args.path

run_config = ClassificationRunConfig(dataset= args.dataset, test_batch_size=args.batch_size, n_worker=args.workers,robust_mode=args.robust_mode)
dyn_network = DYN_net(args.net,args.robust_mode,args.dataset, args.train_criterion ,pretrained=True,run_config=run_config,WPS=args.WPS)
""" Randomly sample a sub-network, 
    you can also manually set the sub-network using: 
        dyn_network.set_active_subnet(ks=7, e=6, d=4) 
"""
# dyn_network.set_active_subnet(ks=3, e=3, d=2)
# dyn_network.set_active_subnet(d=4,e=0.25,w=1) 
import random 
import numpy as np 
random.seed(0)
np.random.seed(0)
acc1,rob1,acc2,rob2 =[],[],[],[]
if args.net == "ResNet50":   
   arch = ResNetArchEncoder(image_size_list=[224 if args.dataset == 'imagenet' else 32],depth_list=[0,1,2],expand_list=[0.2,0.25,0.35],width_mult_list=[0.65,0.8,1.0])
else:
   arch =  MobileNetArchEncoder (image_size_list=[224 if args.dataset == 'imagenet' else 32],depth_list=[2,3,4],expand_list=[3,4,6],ks_list=[3,5,7])
print(arch)
acc_data = AccuracyRobustnessDataset("./acc_rob_data_{}_{}_{}".format(args.dataset,args.net,args.train_criterion))
train_loader, valid_loader, base_acc ,base_rob = acc_data.build_acc_data_loader(arch)
for inputs, targets_acc, targets_rob in train_loader:
    for i in range(len(targets_acc)):
        acc1.append(targets_acc[i].item() * 100)
        rob1.append(targets_rob[i].item() * 100)
    
np.save("./results/acc_mbv3.npy",np.array(acc1))
np.save("./results/rob_mbv3.npy",np.array(rob1))

