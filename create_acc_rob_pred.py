import os
import torch
import argparse
import torch.nn as nn 
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn 
import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *
from proard.classification.data_providers.imagenet import ImagenetDataProvider
from proard.classification.run_manager import DistributedClassificationRunConfig, DistributedRunManager
from proard.model_zoo import DYN_net
from proard.nas.accuracy_predictor import AccuracyDataset,AccuracyPredictor,ResNetArchEncoder,RobustnessPredictor,MobileNetArchEncoder,AccuracyRobustnessDataset,Accuracy_Robustness_Predictor
parser = argparse.ArgumentParser()


def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))
def train(
  model: nn.Module,
  dataloader: DataLoader,
  criterion: nn.Module,
  optimizer: Optimizer,
  callbacks = None,
  epochs = 10,
  save_path = None
) -> None:
  model.cuda()
  model.train()
  for epoch in range(epochs):
    print(epoch)
    for inputs, targets_acc, targets_rob in tqdm(dataloader, desc='train', leave=False):
      inputs = inputs.float().cuda()
      targets_acc = targets_acc.cuda()
      targets_rob = targets_rob.cuda()

      # Reset the gradients (from the last iteration)
      optimizer.zero_grad()

      # Forward inference
      outputs = model(inputs)
      loss = criterion(outputs[:,0], targets_acc) + criterion(outputs[:,1], targets_rob)

      # Backward propagation
      loss.backward()

      # Update optimizer and LR scheduler
      optimizer.step()
      # scheduler.step(epoch)

      if callbacks is not None:
          for callback in callbacks:
              callback()        
  torch.save(model.state_dict(), save_path)
  return model

@torch.inference_mode()
def evaluate(
  model: nn.Module,
  dataloader: DataLoader,
) -> float:
  model.eval()

  for inputs, targets_acc, targets_rob in tqdm(dataloader, desc="eval", leave=False):
    # Move the data from CPU to GPU
    inputs = inputs.cuda()

    targets_acc = targets_acc.cuda()
    targets_rob = targets_rob.cuda()


    # Inference
    outputs = model(inputs)

    # Convert logits to class indices
    print(RMSELoss(outputs[:,0],targets_acc),RMSELoss(outputs[:,1],targets_rob))
  return RMSELoss(outputs[:,0],targets_acc) + RMSELoss(outputs[:,1],targets_rob)


def get_model_flops(model, inputs):
    num_macs = profile_macs(model, inputs)
    return num_macs


def get_model_size(model: nn.Module, data_width=32):
    """
    calculate the model size in bits
    :param data_width: #bits per element
    """
    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()
    return num_elements * data_width





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
    ],
    help="Dyanmic networks",
)
parser.add_argument(
    "--dataset", type=str, default="cifar10" ,choices=["cifar10", "cifar100", "imagenet"]
)
parser.add_argument("--train_criterion", type=str, default="trades",choices=["trades","sat","mart","hat"])
parser.add_argument(
    "--robust_mode", type=bool, default=True
)
args = parser.parse_args()
if args.net == "ResNet50":   
   arch = ResNetArchEncoder(image_size_list=[224 if args.dataset == 'imagenet' else 32],depth_list=[0,1,2],expand_list=[0.2,0.25,0.35],width_mult_list=[0.65,0.8,1.0])
else:
   arch =  MobileNetArchEncoder (image_size_list=[224 if args.dataset == 'imagenet' else 32],depth_list=[2,3,4],expand_list=[3,4,6],ks_list=[3,5,7])
print(arch)
acc_data = AccuracyRobustnessDataset("./acc_rob_data_{}_{}_{}".format(args.dataset,args.net,args.train_criterion))
train_loader, valid_loader, base_acc ,base_rob = acc_data.build_acc_data_loader(arch)
acc_pred_network = Accuracy_Robustness_Predictor(arch_encoder=arch,base_acc_val=None)
# optimizer_ = torch.optim.Adam(acc_pred_network.parameters(),lr=1e-3,weight_decay=1e-4)
# criterion = nn.MSELoss()
# acc_pred_network = train(acc_pred_network,train_loader,criterion,optimizer_,callbacks=None, epochs=50,save_path ="./acc_rob_data_{}_{}_{}/src/model_acc_rob.pth".format(args.dataset,args.net,args.train_criterion).format(args.dataset))
acc_pred_network.load_state_dict(torch.load("./acc_rob_data_{}_{}_{}/src/model_acc_rob.pth".format(args.dataset,args.net,args.train_criterion)))
print(evaluate(acc_pred_network,valid_loader))



# import numpy as np 
# accs=[]
# robs=[]
# pred_accs=[]
# pred_robs=[]
# for x,acc,rob, in valid_loader:
#    for ac in acc:
#       accs.append(ac.item()*100)
#    for ro in rob:
#       robs.append(ro.item()*100)  

# for x,acc,rob, in valid_loader:
#    for arch in x:
#       acc ,rob = acc_pred_network(arch.cuda())
#       pred_accs.append(acc.item()*100)
#       pred_robs.append(rob.item()*100)
# print(accs,robs)      
# print(pred_accs,pred_robs)   
# np.savetxt("./results/accs.csv", np.array(accs), delimiter=",")  
# np.savetxt("./results/robs.csv", np.array(robs), delimiter=",") 
# np.savetxt("./results/pred_accs.csv", np.array(pred_accs), delimiter=",")  
# np.savetxt("./results/pred_robs.csv", np.array(pred_robs), delimiter=",")


