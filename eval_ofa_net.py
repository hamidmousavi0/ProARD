# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import torch
import argparse

from proard.classification.data_providers.imagenet import ImagenetDataProvider
from proard.classification.data_providers.cifar10 import Cifar10DataProvider
from proard.classification.data_providers.cifar100 import Cifar100DataProvider
from proard.classification.run_manager import ClassificationRunConfig, RunManager
from proard.model_zoo import DYN_net


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
    default=16,
)
parser.add_argument("-j", "--workers", help="Number of workers", type=int, default=20)
parser.add_argument(
    "-n",
    "--net",
    metavar="DYNET",
    default="ResNet50",
    choices=[
        "ResNet50",
        "MBV3",
        "ProxylessNASNet",
        "MBV2",
        "WideResNet"
    ],
    help="dynamic networks",
)
parser.add_argument(
    "--dataset", type=str, default="cifar10" ,choices=["cifar10", "cifar100", "imagenet"]
)
parser.add_argument(
    "--attack", type=str, default="autoattack" ,choices=['fgsm', 'linf-pgd', 'fgm', 'l2-pgd', 'linf-df', 'l2-df', 'linf-apgd', 'l2-apgd','squar_attack','autoattack','apgd_ce']
)
parser.add_argument("--train_criterion", type=str, default="trades",choices=["trades","sat","mart","hat"])
parser.add_argument(
    "--robust_mode", type=bool, default=True
)
parser.add_argument(
    "--WPS", type=bool, default=False
)
parser.add_argument(
    "--base", type=bool, default=False
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

run_config = ClassificationRunConfig(attack_type=args.attack,dataset= args.dataset, test_batch_size=args.batch_size, n_worker=args.workers,robust_mode=args.robust_mode)
dyn_network = DYN_net(args.net,args.robust_mode,args.dataset, args.train_criterion ,pretrained=True,run_config=run_config,WPS=args.WPS,base=args.base)
""" Randomly sample a sub-network, 
    you can also manually set the sub-network using: 
        dyn_network.set_active_subnet(ks=7, e=6, d=4) 
"""
if not args.base:
    # dyn_network.set_active_subnet(ks=3, e=4, d=2)
    dyn_network.set_active_subnet(d=2,e=0.35,w=1.0) 
    # dyn_network.sample_active_subnet()
    # dyn_network.set_max_net()
    subnet = dyn_network.get_active_subnet(preserve_weight=True)
    # print(subnet)
else:
    subnet = dyn_network    
""" Test sampled subnet 
"""
run_manager = RunManager(".tmp/eval_subnet", subnet, run_config, init=False)
run_config.data_provider.assign_active_img_size(32)
run_manager.reset_running_statistics(net=subnet)

print("Test random subnet:")
# print(subnet.module_str)

loss, (top1, top5,robust1,robust5) = run_manager.validate(net=subnet,is_test=True)
print("Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f,\t robust1=%.1f,\t robust5=%.1f" % (loss, top1, top5,robust1,robust5))
