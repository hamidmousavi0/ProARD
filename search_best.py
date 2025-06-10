import os
import torch
import argparse
import torch.nn as nn 
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn 
import torch
import random
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *
from proard.model_zoo import DYN_net
from proard.nas.accuracy_predictor import AccuracyPredictor,ResNetArchEncoder,RobustnessPredictor,MobileNetArchEncoder,Accuracy_Robustness_Predictor
from proard.nas.efficiency_predictor import ResNet50FLOPsModel,Mbv3FLOPsModel,ProxylessNASFLOPsModel
from proard.nas.search_algorithm import EvolutionFinder,DynIndividual_mbv,DynIndividual_res,DynRandomSampler,DynProblem_mbv,DynProblem_res,DynSampling,individual_to_arch_res,individual_to_arch_mbv
from utils.profile import trainable_param_num
from pymoo.core.individual import Individual
from pymoo.core.mutation import Mutation
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.variable import Choice
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.mutation.rm import ChoiceRandomMutation
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.core.callback import Callback
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output
from proard.classification.run_manager import ClassificationRunConfig, RunManager
parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--path", help="The path of cifar10", type=str, default="/dataset/cifar10"
)
parser.add_argument("-g", "--gpu", help="The gpu(s) to use", type=str, default="all")
parser.add_argument(
    "-b",
    "--batch-size",
    help="The batch on every device for validation",
    type=int,
    default=100,
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
    help="dynamic networks",
)
parser.add_argument(
    "--dataset", type=str, default="cifar10" ,choices=["cifar10", "cifar100", "imagenet"]
)
parser.add_argument(
    "--attack", type=str, default="linf-pgd" ,choices=['fgsm', 'linf-pgd', 'fgm', 'l2-pgd', 'linf-df', 'l2-df', 'linf-apgd', 'l2-apgd','squar_attack','autoattack','apgd_ce']
)
parser.add_argument("--train_criterion", type=str, default="trades",choices=["trades","sat","mart","hat"])
parser.add_argument(
    "--robust_mode", type=bool, default=True
)
args = parser.parse_args()
if args.gpu == "all":
    device_list = range(torch.cuda.device_count())
    args.gpu = ",".join(str(_) for _ in device_list)
else:
    device_list = [int(_) for _ in args.gpu.split(",")]
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.batch_size = args.batch_size * max(len(device_list), 1)
run_config = ClassificationRunConfig(attack_type=args.attack, dataset= args.dataset, test_batch_size=args.batch_size, n_worker=args.workers,robust_mode=args.robust_mode)
dyn_network = DYN_net(args.net,args.robust_mode,args.dataset,args.train_criterion, pretrained=True,run_config=run_config)
if args.net == "ResNet50":
    efficiency_predictor = ResNet50FLOPsModel(dyn_network)
    arch = ResNetArchEncoder(image_size_list=[32],depth_list=[0,1,2],expand_list=[0.2,0.25,0.35],width_mult_list=[0.65,0.8,1.0])
    accuracy_robustness_predictor = Accuracy_Robustness_Predictor(arch)
    accuracy_robustness_predictor.load_state_dict(torch.load("./acc_rob_data_{}_{}_{}/src/model_acc_rob.pth".format(args.dataset,args.net,args.train_criterion)))
elif args.net == "MBV3":
    efficiency_predictor = Mbv3FLOPsModel(dyn_network)
    arch = MobileNetArchEncoder(image_size_list=[32],depth_list=[2,3,4],expand_list=[3,4,6],ks_list=[3,5,7])
    accuracy_robustness_predictor = Accuracy_Robustness_Predictor(arch)
    accuracy_robustness_predictor.load_state_dict(torch.load("./acc_rob_data_{}_{}_{}/src/model_acc_rob.pth".format(args.dataset,args.net,args.train_criterion)))
elif args.net == "ProxylessNASNet":
    efficiency_predictor = ProxylessNASFLOPsModel(dyn_network)
    arch = MobileNetArchEncoder(image_size_list=[32],depth_list=[2,3,4],expand_list=[3,4,6],width_mult_list=[3,5,7])
    accuracy_robustness_predictor = Accuracy_Robustness_Predictor(arch)
    accuracy_robustness_predictor.load_state_dict(torch.load("./acc_rob_data_{}_{}_{}/src/model_acc_rob.pth".format(args.dataset,args.net,args.train_criterion)))
##### Test #################################################
dyn_sampler = DynRandomSampler(arch, efficiency_predictor)
# arch1, eff1 = dyn_sampler.random_sample()
# arch2, eff2 = dyn_sampler.random_sample()
# print(accuracy_predictor.predict_acc([arch1, arch2]))
# print(arch1,eff1)
##################################################

""" Hyperparameters
- P: size of the population in each generation (number of individuals)
- N: number of generations to run the algorithm
- mutate_prob: probability of gene mutation in the evolutionary search
"""
P = 100
N = 100
mutation_prob = 0.5


# variables options
if args.net == 'ResNet50':
    search_space = {
        'e': [0.2, 0.25, 0.35],
        'd': [0, 1, 2],
        'w': [0 ,1 ,2],
        'image_size': [32]
    }
else:
    search_space = {
        'ks': [3, 5, 7],
        'e': [3, 4, 6],
        'd': [2, 3, 4],
        'image_size': [32]
    }

#----------------------------
# units
num_blocks = arch.max_n_blocks
num_stages = arch.n_stage
Flops_constraints = 1600
if args.net == "ResNet50":
    problem = DynProblem_res(efficiency_predictor, accuracy_robustness_predictor, num_blocks, num_stages, search_space,Flops_constraints)
else:
    problem = DynProblem_mbv(efficiency_predictor, accuracy_robustness_predictor, num_blocks, num_stages, search_space,Flops_constraints)





mutation_rc = ChoiceRandomMutation(prob=1.0, prob_var=0.1)
crossover_ux = UniformCrossover(prob=1.0)
# selection_tournament = TournamentSelection(
#     func_comp=accuracy_predictor.predict_acc, 
#     pressure=2
# )
termination_default = DefaultMultiObjectiveTermination(
    xtol=1e-8, cvtol=1e-6, ftol=0.0025, period=30, n_max_gen=1000, n_max_evals=100000
)
termination_gen = get_termination("n_gen", N)
np.random.seed(42)
random.seed(42)
if args.net=="ResNet50":
    init_pop = Population(individuals=[DynIndividual_res(dyn_sampler.random_sample(), accuracy_robustness_predictor) for _ in range(P)])
else:
    init_pop = Population(individuals=[DynIndividual_mbv(dyn_sampler.random_sample(), accuracy_robustness_predictor) for _ in range(P)])

algorithm = NSGA2(
        pop_size=P,
        sampling=DynSampling(),
        # selection=selection_tournament,
        crossover=crossover_ux,
        mutation=mutation_rc,
        # mutation=mutation_pm,
        # survival=RankAndCrowdingSurvival(),
        # output=MultiObjectiveOutput(),
        #    **kwargs
    )
res_nsga2 = minimize(
    problem,
    algorithm,
    termination=termination_gen,
    seed=1,
    #verbose=True,
    verbose=False,
    save_history=True,
)
# print(100-res_nsga2.history[99].pop.get('F')[:,0],100-res_nsga2.history[99].pop.get('F')[:,1])
# a = individual_to_arch_res(res_nsga2.pop.get('X'),num_blocks)[0]
# # print(a)
# # a['d'][3]  = int(a['d'][3])
# a['d'][4]  = int(a['d'][4])
# dyn_network.set_active_subnet(**a)
# subnet = dyn_network.get_active_subnet(preserve_weight=True)
# run_manager = RunManager(".tmp/eval_subnet", subnet, run_config, init=False)
# run_config.data_provider.assign_active_img_size(32)
# run_manager.reset_running_statistics(net=subnet)

# print("Test random subnet:")
# # print(subnet.module_str)

# loss, (top1, top5,robust1,robust5) = run_manager.validate(net=subnet,is_test=True)
# print("Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f,\t robust1=%.1f,\t robust5=%.1f" % (loss, top1, top5,robust1,robust5))


np.savetxt("./results/acc_gen0.csv", 100-res_nsga2.history[0].pop.get('F')[:,0], delimiter=",") 

np.savetxt("./results/acc_gen99.csv", 100-res_nsga2.history[99].pop.get('F')[:,0], delimiter=",")   
np.savetxt("./results/rob_gen0.csv", 100-res_nsga2.history[0].pop.get('F')[:,1], delimiter=",") 

np.savetxt("./results/rob_gen99.csv", 100-res_nsga2.history[99].pop.get('F')[:,1], delimiter=",")
np.savetxt("./results/flops_gen99.csv", res_nsga2.history[99].pop.get('G'), delimiter=",")

# np.savetxt("./results/robs.csv", np.array(robs), delimiter=",") 

from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
# NSGA-II population progression
x_min, x_max, y_min, y_max = 80, 93, 47, 56
ax_limits = [x_min, x_max, y_min, y_max]
#-------------------------------------------------
# plot
fig, ax = plt.subplots(dpi=600)
gen0 = 0
gen1 = 99
print(100-res_nsga2.history[gen1].pop.get('F')[:,0], 100 - res_nsga2.history[gen1].pop.get('F')[:,1])
# gen2 = 99
# print(res_nsga2.history[gen0].pop.get('F')[:,0],res_nsga2.history[gen0].pop.get('F')[:,1]  )
ax.plot(100-res_nsga2.history[gen0].pop.get('F')[:,0], 100 - res_nsga2.history[gen0].pop.get('F')[:,1]  , 'o', label=f'Population at generation #{gen0+1}', color='red',    alpha=0.5)
ax.plot(100-res_nsga2.history[gen1].pop.get('F')[:,0], 100 - res_nsga2.history[gen1].pop.get('F')[:,1] , 'o', label=f'Population at generation #{gen1+1}', color='green',  alpha=0.5)
# ax.plot(res_nsga2.history[gen2].pop.get('F')[:,0], 100 - res_nsga2.history[gen2].pop.get('F')[:,1], 'o', label=f'Population at generation #{gen2+1}', color='orange', alpha=0.5)
# ax.plot(res_nsga2.history[gen3].pop.get('F')[:,0], 100 - res_nsga2.history[gen3].pop.get('F')[:,1], 'o', label=f'Population at generation #{gen3+1}', color='blue',   alpha=0.5)
#-------------------------------------------------
# text
ax.grid(True, linestyle=':')
ax.set_xlabel('Accuracy (%)')
ax.set_ylabel('Robustness (%)')
ax.set_title('NSGA-II solutions progression For Fixed number of FLOPs'),
ax.legend()
#-------------------------------------------------
# x-axis
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(1))
# y-axis
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(1))
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.set(xlim=(ax_limits[0], ax_limits[1]), ylim=(ax_limits[2], ax_limits[3]))
#-------------------------------------------------
plt.savefig('nsga2_pop_progression_debug.png')
fig.set_dpi(100)
# plt.close(fig)



# plt.show()


# finder  = EvolutionFinder(efficiency_predictor,accuracy_predictor,Robustness_predictor)
# valid_constraint_range = 800
# best_valids, best_info = finder.run_evolution_search(constraint=valid_constraint_range,verbose=True)
# print(efficiency_predictor.get_efficiency(best_info[2]))
# dyn_network.set_active_subnet(best_info[2]['d'],best_info[2]['e'],best_info[2]['w'])
# subnet = dyn_network.get_active_subnet(preserve_weight=True)
# run_config = CifarRunConfig_robust(test_batch_size=args.batch_size, n_worker=args.workers)
# run_manager = RunManager_robust(".tmp/eval_subnet", subnet, run_config, init=False)
# run_config.data_provider.assign_active_img_size(32)
# run_manager.reset_running_statistics(net=subnet)
# loss, (top1, top5,robust1,robust5) = run_manager.validate(net=subnet)
# print("Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f,\t robust1=%.1f,\t robust5=%.1f" % (loss, top1, top5,robust1,robust5))
# print("number of parameter={}M".format(trainable_param_num(subnet)))