# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

from .proxyless_nets import *
from .mobilenet_v3 import *
from .resnets import *
from .wide_resnet import WideResNet
from .resnet_trades import *

def get_net_by_name(name):
    if name == ProxylessNASNets.__name__:
        return ProxylessNASNets
    elif name == MobileNetV3.__name__:
        return MobileNetV3
    elif name == ResNets.__name__:
        return ResNets
    if name == ProxylessNASNets_Cifar.__name__:
        return ProxylessNASNets_Cifar
    elif name == MobileNetV3_Cifar.__name__:
        return MobileNetV3
    elif name == ResNets_Cifar.__name__:
        return ResNets_Cifar
    else:
        raise ValueError("unrecognized type of network: %s" % name)
