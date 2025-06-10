import numpy as np
from pymoo.core.individual import Individual
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.variable import Choice
__all__ = ["individual_to_arch_mbv","DynIndividual_mbv","DynProblem_mbv","individual_to_arch_res","DynIndividual_res","DynProblem_res","DynSampling","DynRandomSampler"]
def individual_to_arch_mbv(population, n_blocks):
    archs = []
    for individual in population:
        archs.append(
            {
                "ks": individual[0:n_blocks],
                "e": individual[n_blocks : 2 * n_blocks],
                "d": individual[2 * n_blocks : -1],
                "image_size": individual[-1:],
            }
        )    
    return archs
class DynIndividual_mbv(Individual):
    def __init__(self, individual, accuracy_predictor,Robustness_predictor, config=None, **kwargs):
        super().__init__(config=None, **kwargs)
        self.X = np.concatenate(
            (
                individual[0]["ks"],
                individual[0]["e"],
                individual[0]["d"],
                individual[0]["image_size"],
            )
        )
        self.flops = individual[1]
        self.accuracy = 100 - accuracy_predictor.predict_acc([individual[0]])
        self.robustness = 100 - Robustness_predictor.predict_rob([individual[0]])
        self.F = np.concatenate(([self.flops], [self.accuracy.squeeze().cpu().detach().numpy()],[self.robustness.squeeze().cpu().detach().numpy()]))



class DynProblem_mbv(Problem):
    def __init__(self, efficiency_predictor, accuracy_predictor, robustness_predictor, num_blocks, num_stages, search_vars):
        self.ks = Choice(options=search_vars.get('ks'))
        self.e = Choice(options=search_vars.get('e'))
        self.d = Choice(options=search_vars.get('d'))
        self.r = Choice(options=search_vars.get('image_size'))
        
        super().__init__(
            vars= dict(zip(range(len(num_blocks * [self.ks] + num_blocks * [self.e] + num_stages * [self.d] + [self.r])), num_blocks * [self.ks] + num_blocks * [self.e] + num_stages * [self.d] + [self.r])),
            n_obj=3,
            n_constr=0,
        )
        self.efficiency_predictor = efficiency_predictor
        self.accuracy_predictor = accuracy_predictor
        self.robustness_predictor = robustness_predictor
        self.blocks = num_blocks
        self.stages = num_stages
        self.search_vars = search_vars

    def _evaluate(self, x, out, *args, **kwargs):
        f1=[]
        # x.shape = (population_size, n_var) = (100, 4)
        arch = individual_to_arch_mbv(x, self.blocks)
        for arc in arch:
            f1.append(self.efficiency_predictor.get_efficiency(arc))      
        f2 = 100 - self.accuracy_predictor.predict_acc(arch).detach().cpu().numpy()
        f3 = 100 - self.robustness_predictor.predict_rob(arch).detach().cpu().numpy()
        out["F"] = np.column_stack([f1, f2,f3])


def individual_to_arch_res(population, n_blocks):
    archs = []
    for individual in population:
        archs.append(
            {
                "e": individual[n_blocks : 2 * n_blocks],
                "d": individual[2 * n_blocks : -1],
                "w": individual[0:n_blocks],
                "r": individual[-1:],
            }
        )
    return archs
class DynIndividual_res(Individual):
    def __init__(self, individual, accuracy_predictor,Robustness_predictor, config=None, **kwargs):
        super().__init__(config=None, **kwargs)
        self.X = np.concatenate(
            (
                individual[0]["e"],
                individual[0]["d"],
                 individual[0]["w"],
                [individual[0]["image_size"]],
            )
        )
        self.flops = individual[1]
        self.accuracy = 100 - accuracy_predictor.predict_acc([individual[0]])
        self.robustness = 100 - Robustness_predictor.predict_rob([individual[0]])
        self.F = np.concatenate(([self.flops], [self.accuracy.squeeze().cpu().detach().numpy()],[self.robustness.squeeze().cpu().detach().numpy()]))



class DynProblem_res(Problem):
    def __init__(self, efficiency_predictor, accuracy_predictor, robustness_predictor, num_blocks, num_stages, search_vars):
        self.e = Choice(options=search_vars.get('e'))
        self.d = Choice(options=search_vars.get('d'))
        self.w = Choice(options=search_vars.get('w'))
        self.r = Choice(options=search_vars.get('image_size'))
        super().__init__(
            vars= dict(zip(range(len(num_blocks * [self.ks] + num_blocks * [self.e] + num_stages * [self.d] + [self.r])), num_blocks * [self.ks] + num_blocks * [self.e] + num_stages * [self.d] + [self.r])),
            n_obj=3,
            n_constr=0,
        )
        self.efficiency_predictor = efficiency_predictor
        self.accuracy_predictor = accuracy_predictor
        self.robustness_predictor = robustness_predictor
        self.blocks = num_blocks
        self.stages = num_stages
        self.search_vars = search_vars
        
    def _evaluate(self, x, out, *args, **kwargs):
        f1={}
        # x.shape = (population_size, n_var) = (100, 4)
        arch = individual_to_arch_res(x, self.blocks)
        for arc in arch:
            f1.append(self.efficiency_predictor.get_efficiency(arc))
        f2 = 100 - self.accuracy_predictor.predict_acc(arch)
        f3 = 100 - self.robustness_predictor.predict_rob(arch)
        out["F"] = np.column_stack([f1, f2,f3])



class DynSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        return [
            [np.random.choice(var.options) for key,var in problem.vars.items()]
            for _ in range(n_samples)
        ]


class DynRandomSampler:
    def __init__(self, arch_manager, efficiency_predictor):
        self.arch_manager = arch_manager
        self.efficiency_predictor = efficiency_predictor

    def random_sample(self):
        sample = self.arch_manager.random_sample_arch()
        efficiency = self.efficiency_predictor.get_efficiency(sample)
        return sample, efficiency