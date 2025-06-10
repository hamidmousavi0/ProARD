import os
import torch
import argparse

from proard.classification.data_providers.imagenet import ImagenetDataProvider
from proard.classification.run_manager import DistributedClassificationRunConfig, DistributedRunManager
from proard.model_zoo import DYN_net
from proard.nas.accuracy_predictor import AccuracyRobustnessDataset
import horovod.torch as hvd
parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--path", help="The path of cifar10", type=str, default="/dataset/cifar10"
)
parser.add_argument("-g", "--gpu", help="The gpu(s) to use", type=str, default="all")
parser.add_argument(
    "-b",
    "--batch_size",
    help="The batch on every device for validation",
    type=int,
    default=32,
)
parser.add_argument("-j", "--workers", help="Number of workers", type=int, default=20)
parser.add_argument(
    "-n",
    "--net",
    metavar="DYNNET",
    default="ResNet50",
    choices=[
        "ResNet50",
        "MBV3",
        "ProxylessNASNet",
        "MBV2"
    ],
    help="Dynamic networks",
)
parser.add_argument(
    "--dataset", type=str, default="cifar10" ,choices=["cifar10", "cifar100", "imagenet"]
)
parser.add_argument("--train_criterion", type=str, default="trades",choices=["trades","sat","mart","hat"])
parser.add_argument(
    "--robust_mode", type=bool, default=True
)
parser.add_argument(
    "--WPS", type=bool, default=True
)
parser.add_argument(
    "--base", type=bool, default=False
)
# Initialize Horovod
hvd.init()
    # Pin GPU to be used to process local rank (one GPU per process)
torch.cuda.set_device(hvd.local_rank())
num_gpus = hvd.size()

args = parser.parse_args()
if args.gpu == "all":
    device_list = range(torch.cuda.device_count())
    args.gpu = ",".join(str(_) for _ in device_list)
else:
    device_list = [int(_) for _ in args.gpu.split(",")]
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.test_batch_size = args.batch_size # * max(len(device_list), 1)
ImagenetDataProvider.DEFAULT_PATH = args.path


distributed_run_config = DistributedClassificationRunConfig(**args.__dict__, num_replicas=num_gpus, rank=hvd.rank())
dyn_network = DYN_net(args.net, args.robust_mode , args.dataset, args.train_criterion, pretrained=True,run_config=distributed_run_config,WPS=args.WPS)
compression = hvd.Compression.none
distributed_run_manager = DistributedRunManager(".tmp/eval_subnet", dyn_network, distributed_run_config,compression,is_root=(hvd.rank() == 0),init=False)
distributed_run_manager.save_config()
    # hvd broadcast
distributed_run_manager.broadcast()
acc_data = AccuracyRobustnessDataset("./acc_rob_data_WPS_{}_{}_{}".format(args.dataset,args.net,args.train_criterion))

acc_data.build_acc_rob_dataset(distributed_run_manager,dyn_network,image_size_list=[224 if args.dataset == "imagenet" else 32])