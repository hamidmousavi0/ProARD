# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import json
import torch
import gdown

from proard.classification.networks import get_net_by_name, ResNet50
from proard.classification.elastic_nn.networks import (
    DYNResNets,DYNMobileNetV3,DYNProxylessNASNets,DYNProxylessNASNets_Cifar,DYNMobileNetV3_Cifar,DYNResNets_Cifar
)
from proard.classification.networks import (WideResNet,ResNet50_Cifar,ResNet50,MobileNetV3_Cifar,MobileNetV3Large_Cifar,MobileNetV3Large,ProxylessNASNets_Cifar,ProxylessNASNets,MobileNetV2_Cifar,MobileNetV2)
__all__ = [
    "DYN_net",
]



def DYN_net(net_id, robust_mode, dataset,train_criterion, pretrained=True,run_config=None,WPS=False,base=False):
    if net_id == "ResNet50":
        if not base:
            if dataset == "cifar10" or dataset == "cifar100":
                net = DYNResNets_Cifar(n_classes=run_config.data_provider.n_classes,
                    dropout_rate=0,
                    depth_list=[0, 1, 2],
                    expand_ratio_list=[0.2, 0.25, 0.35],
                    width_mult_list=[0.65, 0.8, 1.0],
                )
            else:
                net = DYNResNets(n_classes=run_config.data_provider.n_classes,
                    dropout_rate=0,
                    depth_list=[0, 1, 2],
                    expand_ratio_list=[0.2, 0.25, 0.35],
                    width_mult_list=[0.65, 0.8, 1.0],
                ) 
        else:
            if dataset == "cifar10" or dataset == "cifar100":
                net = ResNet50_Cifar(n_classes=run_config.data_provider.n_classes)
            else:
                net = ResNet50(n_classes=run_config.data_provider.n_classes) 

    elif net_id == "MBV3":
        if not base:
            if dataset == "cifar10" or dataset == "cifar100":
                net = DYNMobileNetV3_Cifar(n_classes=run_config.data_provider.n_classes,
                    dropout_rate=0,
                    width_mult=1.0,
                    ks_list=[3, 5, 7],
                    expand_ratio_list=[3, 4, 6],
                    depth_list=[2, 3, 4],
                )
            else:
                net = DYNMobileNetV3(n_classes=run_config.data_provider.n_classes,
                    dropout_rate=0,
                    width_mult=1.0,
                    ks_list=[3, 5, 7],
                    expand_ratio_list=[3, 4, 6],
                    depth_list=[2, 3, 4],
                ) 
        else:
            if dataset == "cifar10" or dataset == "cifar100":
                net = MobileNetV3Large_Cifar(n_classes=run_config.data_provider.n_classes)
            else:
                net = MobileNetV3Large(n_classes=run_config.data_provider.n_classes)         
    elif net_id == "ProxylessNASNet":
        if not base:
            if dataset == "cifar10" or dataset == "cifar100":
                net = DYNProxylessNASNets_Cifar(n_classes=run_config.data_provider.n_classes,
                    dropout_rate=0,
                    width_mult=1.0,
                    ks_list=[3, 5, 7],
                    expand_ratio_list=[3, 4, 6],
                    depth_list=[2, 3, 4],
                )
            else:
                net = DYNProxylessNASNets(n_classes=run_config.data_provider.n_classes,
                    dropout_rate=0,
                    width_mult=1.0,
                    ks_list=[3, 5, 7],
                    expand_ratio_list=[3, 4, 6],
                    depth_list=[2, 3, 4],
                )     
        else:
            if dataset == "cifar10" or dataset == "cifar100":
                net = ProxylessNASNets_Cifar(n_classes=run_config.data_provider.n_classes)
            else:
                net = ProxylessNASNets(n_classes=run_config.data_provider.n_classes)    
    elif net_id == "MBV2":
        if not base:
            if dataset == "cifar10" or dataset == "cifar100":
                net = DYNProxylessNASNets_Cifar(n_classes=run_config.data_provider.n_classes,
                    dropout_rate=0,
                    base_stage_width="google",
                    width_mult=1.0,
                    ks_list=[3, 5, 7],
                    expand_ratio_list=[3, 4, 6],
                    depth_list=[2, 3, 4],
                )
            else:
                net = DYNProxylessNASNets(n_classes=run_config.data_provider.n_classes,
                    dropout_rate=0,
                    base_stage_width="google",
                    width_mult=1.0,
                    ks_list=[3, 5, 7],
                    expand_ratio_list=[3, 4, 6],
                    depth_list=[2, 3, 4],
                )   
        else:
            if dataset == "cifar10" or dataset == "cifar100":
                net = MobileNetV2_Cifar(n_classes=run_config.data_provider.n_classes)
            else:
                net = MobileNetV2(n_classes=run_config.data_provider.n_classes)     
    elif net_id == "WideResNet": 
        if dataset == "cifar10" or dataset == "cifar100":
            net = WideResNet(num_classes=run_config.data_provider.n_classes)
        else:
            raise ValueError("Not supported: %s" % net_id)

    else:
        raise ValueError("Not supported: %s" % net_id)

    if pretrained and not WPS and not base:
        if net_id == "ResNet50":
            if robust_mode:
                pt_path = "exp/robust/"+ dataset + "/" + net_id + '/' + train_criterion +"/width_depth2width_depth_width/phase2" + "/checkpoint/model_best.pth.tar"
            else:
                pt_path = "exp/"+ dataset + "/" + net_id + '/' + train_criterion + "/width_depth2width_depth_width/phase2" + "/checkpoint/model_best.pth.tar"      
        else:
            if robust_mode:
                pt_path = "exp/robust/"+ dataset + '/' + net_id + '/' + train_criterion +"/kernel_depth2kernel_depth_width/phase2" +  "/checkpoint/model_best.pth.tar" 
                
            else:
                pt_path = "exp/"+ dataset + '/' + net_id + '/' + train_criterion +"/kernel_depth2kernel_depth_width/phase2" +  "/checkpoint/model_best.pth.tar"    
    elif  pretrained and WPS and not base:  
        if net_id == "ResNet50":
            if robust_mode:
                pt_path = "exp/robust/WPS/"+ dataset + "/" + net_id + '/' + train_criterion +"/width_depth2width_depth_width/phase2" + "/checkpoint/model_best.pth.tar"
            else:
                pt_path = "exp/WPS/"+ dataset + "/" + net_id + '/' + train_criterion + "/width_depth2width_depth_width/phase2" + "/checkpoint/model_best.pth.tar"      
        else:
            if robust_mode:
                pt_path = "exp/robust/WPS/"+ dataset + '/' + net_id + '/' + train_criterion +"/kernel_depth2kernel_depth_width/phase2" +  "/checkpoint/model_best.pth.tar" 
                
            else:
                pt_path = "exp/WPS/"+ dataset + '/' + net_id + '/' + train_criterion +"/kernel_depth2kernel_depth_width/phase2" +  "/checkpoint/model_best.pth.tar"           
    else:
        if not base:
            pt_path = "exp/robust/teacher/"+ dataset + '/' + net_id + '/' + train_criterion +  "/checkpoint/model_best.pth.tar"     
        else:
            pt_path = "exp/robust/base/"+ dataset + '/' + net_id + '/' + train_criterion +  "/checkpoint/model_best.pth.tar"    
    print(pt_path)                 
    init = torch.load(pt_path, map_location="cuda")["state_dict"]
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in init.items():
    #     name = k[7:] # remove `module.`
    #     new_state_dict[name] = v
    net.load_state_dict(init)
    return net


