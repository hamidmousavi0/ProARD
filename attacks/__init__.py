from .base import Attack


from .fgsm import FGMAttack
from .fgsm import FGSMAttack
from .fgsm import L2FastGradientAttack
from .fgsm import LinfFastGradientAttack

from .pgd import PGDAttack
from .pgd import L2PGDAttack
from .pgd import LinfPGDAttack

from .deepfool import DeepFoolAttack
from .deepfool import LinfDeepFoolAttack
from .deepfool import L2DeepFoolAttack

from .utils import CWLoss
from .autoattack import AutoAttacks
from .apgd_ce import  Autoattack_apgd_ce
from .squred import  Squre_Attack


ATTACKS = ['fgsm', 'linf-pgd', 'fgm', 'l2-pgd', 'linf-df', 'l2-df', 'linf-apgd', 'l2-apgd','squar_attack','autoattack','apgd_ce']


def create_attack(model, criterion, attack_type, attack_eps, attack_iter, attack_step, rand_init_type='uniform', 
                  clip_min=0., clip_max=1.):
    """
    Initialize adversary.
    Arguments:
        model (nn.Module): forward pass function.
        criterion (nn.Module): loss function.
        attack_type (str): name of the attack.
        attack_eps (float): attack radius.
        attack_iter (int): number of attack iterations.
        attack_step (float): step size for the attack.
        rand_init_type (str): random initialization type for PGD (default: uniform).
        clip_min (float): mininum value per input dimension.
        clip_max (float): maximum value per input dimension.
   Returns:
       Attack
   """
    
    if attack_type == 'fgsm':
        attack = FGSMAttack(model, criterion, eps=attack_eps, clip_min=clip_min, clip_max=clip_max)
    elif attack_type == 'fgm':
        attack = FGMAttack(model, criterion, eps=attack_eps, clip_min=clip_min, clip_max=clip_max)
    elif attack_type == 'linf-pgd':
        attack = LinfPGDAttack(model, criterion, eps=attack_eps, nb_iter=attack_iter, eps_iter=attack_step,
                               rand_init_type=rand_init_type, clip_min=clip_min, clip_max=clip_max)
    elif attack_type == 'l2-pgd':
        attack = L2PGDAttack(model, criterion, eps=attack_eps, nb_iter=attack_iter, eps_iter=attack_step, 
                             rand_init_type=rand_init_type, clip_min=clip_min, clip_max=clip_max)
    elif attack_type == 'linf-df':
        attack = LinfDeepFoolAttack(model, overshoot=0.02, nb_iter=attack_iter, search_iter=0, clip_min=clip_min, 
                                    clip_max=clip_max)
    elif attack_type == 'l2-df':
        attack = L2DeepFoolAttack(model, overshoot=0.02, nb_iter=attack_iter, search_iter=0, clip_min=clip_min, 
                                  clip_max=clip_max)
    elif  attack_type == 'squar_attack': 
        attack =  Squre_Attack(model, criterion, nb_iter=attack_iter, eps_iter=attack_step, 
                             rand_init_type=rand_init_type, clip_min=clip_min, clip_max=clip_max) 
    elif attack_type == "autoattack":
        attack =  AutoAttacks(model, nb_iter=attack_iter, eps=attack_eps, eps_iter=attack_step, 
                             rand_init_type=rand_init_type, clip_min=clip_min, clip_max=clip_max) 
    elif attack_type == "apgd_ce":
        attack = Autoattack_apgd_ce (model, nb_iter=attack_iter, eps_iter=attack_step, 
                             rand_init_type=rand_init_type, clip_min=clip_min, clip_max=clip_max)
    else:
        raise NotImplementedError('{} is not yet implemented!'.format(attack_type))
    return attack