# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

from proard.utils import calc_learning_rate, build_optimizer
from proard.classification.data_providers import ImagenetDataProvider
from proard.classification.data_providers import Cifar10DataProvider
from proard.classification.data_providers import Cifar100DataProvider
from robust_loss.trades import trades_loss
from robust_loss.adaad import adaad_loss
from robust_loss.ard import ard_loss
from robust_loss.hat import hat_loss
from robust_loss.mart import mart_loss
from robust_loss.sat import sat_loss
from robust_loss.rslad import rslad_loss
import torch
__all__ = ["RunConfig", "ClassificationRunConfig", "DistributedClassificationRunConfig"]


class RunConfig:
    def __init__(
        self,
        n_epochs,
        init_lr,
        lr_schedule_type,
        lr_schedule_param,
        dataset,
        train_batch_size,
        test_batch_size,
        valid_size,
        opt_type,
        opt_param,
        weight_decay,
        label_smoothing,
        no_decay_keys,
        mixup_alpha,
        model_init,
        validation_frequency,
        print_frequency,
    ):
        self.n_epochs = n_epochs
        self.init_lr = init_lr
        self.lr_schedule_type = lr_schedule_type
        self.lr_schedule_param = lr_schedule_param

        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.valid_size = valid_size

        self.opt_type = opt_type
        self.opt_param = opt_param
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.no_decay_keys = no_decay_keys

        self.mixup_alpha = mixup_alpha

        self.model_init = model_init
        self.validation_frequency = validation_frequency
        self.print_frequency = print_frequency
       
    @property
    def config(self):
        config = {}
        for key in self.__dict__:
            if not key.startswith("_"):
                config[key] = self.__dict__[key]
        return config

    def copy(self):
        return RunConfig(**self.config)

    """ learning rate """

    def adjust_learning_rate(self, optimizer, epoch, batch=0, nBatch=None):
        """adjust learning of a given optimizer and return the new learning rate"""
        new_lr = calc_learning_rate(
            epoch, self.init_lr, self.n_epochs, batch, nBatch, self.lr_schedule_type
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr

    def warmup_adjust_learning_rate(
        self, optimizer, T_total, nBatch, epoch, batch=0, warmup_lr=0
    ):
        T_cur = epoch * nBatch + batch + 1
        new_lr = T_cur / T_total * (self.init_lr - warmup_lr) + warmup_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr

    """ data provider """

    @property
    def data_provider(self):
        raise NotImplementedError

    @property
    def train_loader(self):
        return self.data_provider.train

    @property
    def valid_loader(self):
        return self.data_provider.valid

    @property
    def test_loader(self):
        return self.data_provider.test

    def random_sub_train_loader(
        self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None
    ):
        return self.data_provider.build_sub_train_loader(
            n_images, batch_size, num_worker, num_replicas, rank
        )

    """ optimizer """

    def build_optimizer(self, net_params):
        return build_optimizer(
            net_params,
            self.opt_type,
            self.opt_param,
            self.init_lr,
            self.weight_decay,
            self.no_decay_keys,
        )



class ClassificationRunConfig(RunConfig):
    def __init__(
        self,
        n_epochs=150,
        init_lr=0.05,
        lr_schedule_type="cosine",
        lr_schedule_param=None,
        dataset="imagenet", # 'cifar10' or 'cifar100'
        train_batch_size=256,
        test_batch_size=500,
        valid_size=None,
        opt_type="sgd",
        opt_param=None,
        weight_decay=4e-5,
        label_smoothing=0.1,
        no_decay_keys=None,
        mixup_alpha=None,
        model_init="he_fout",
        validation_frequency=1,
        print_frequency=10,
        n_worker=32,
        resize_scale=0.08,
        distort_color="tf",
        image_size=224, # 32
        robust_mode = False,
        epsilon_train = 0.031,
        num_steps_train = 10,
        step_size_train = 0.0078,
        clip_min_train  = 0 ,
        clip_max_train = 1,
        const_init_train = False,
        beta_train = 6.0,
        distance_train ="l_inf",
        epsilon_test = 0.031,
        num_steps_test = 20,
        step_size_test = 0.0078,
        clip_min_test = 0,
        clip_max_test = 1,
        const_init_test = False,
        beta_test = 6.0,
        distance_test = "l_inf",
        train_criterion = "trades",
        test_criterion = "ce",
        kd_criterion = 'rslad',
        attack_type = "linf-pgd",
        **kwargs
    ):
        super(ClassificationRunConfig, self).__init__(
            n_epochs,
            init_lr,
            lr_schedule_type,
            lr_schedule_param,
            dataset,
            train_batch_size,
            test_batch_size,
            valid_size,
            opt_type,
            opt_param,
            weight_decay,
            label_smoothing,
            no_decay_keys,
            mixup_alpha,
            model_init,
            validation_frequency,
            print_frequency,
        )

        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color
        self.image_size = image_size
        self.epsilon_train = epsilon_train
        self.num_steps_train = num_steps_train
        self.step_size_train  = step_size_train
        self.clip_min_train = clip_min_train
        self.clip_max_train = clip_max_train
        self.const_init_train = const_init_train
        self.beta_train = beta_train
        self.distance_train =  distance_train
        self.epsilon_test = epsilon_test
        self.num_steps_test = num_steps_test
        self.step_size_test  = step_size_test
        self.clip_min_test = clip_min_test
        self.clip_max_test = clip_max_test
        self.const_init_test = const_init_test
        self.beta_test = beta_test
        self.distance_test =  distance_test
        self.train_criterion = train_criterion
        self.test_criterion = test_criterion
        self.kd_criterion = kd_criterion
        self.attack_type = attack_type
        self.robust_mode = robust_mode
    @property
    def data_provider(self):
        if self.__dict__.get("_data_provider", None) is None:
            if self.dataset == ImagenetDataProvider.name():
                DataProviderClass = ImagenetDataProvider
            elif self.dataset == Cifar10DataProvider.name():   
                DataProviderClass = Cifar10DataProvider
            elif self.dataset == Cifar100DataProvider.name():   
                DataProviderClass = Cifar100DataProvider        
            else:
                raise NotImplementedError
            self.__dict__["_data_provider"] = DataProviderClass(
                train_batch_size=self.train_batch_size,
                test_batch_size=self.test_batch_size,
                valid_size=self.valid_size,
                n_worker=self.n_worker,
                resize_scale=self.resize_scale,
                distort_color=self.distort_color,
                image_size=self.image_size,
            )
        return self.__dict__["_data_provider"]
    @property
    def train_criterion_loss (self):
        if self.train_criterion == "trades" : 
            return trades_loss
        elif self.train_criterion == "mart" : 
            return mart_loss 
        elif self.train_criterion == "sat"  :
            return sat_loss
        elif self.train_criterion == "hat" : 
            return hat_loss
    @property
    def test_criterion_loss (self) : 
        if self.test_criterion == "ce" : 
            return torch.nn.CrossEntropyLoss()     
    @property 
    def kd_criterion_loss (self) : 
        if self.kd_criterion =="ard" :
            return ard_loss
        elif self.kd_criterion == "adaad" : 
            return adaad_loss 
        elif self.kd_criterion == "rslad" : 
            return   rslad_loss 
class DistributedClassificationRunConfig(ClassificationRunConfig):
    def __init__(
        self,
        n_epochs=150,
        init_lr=0.05,
        lr_schedule_type="cosine",
        lr_schedule_param=None,
        dataset="imagenet",
        train_batch_size=64,
        test_batch_size=64,
        valid_size=None,
        opt_type="sgd",
        opt_param=None,
        weight_decay=4e-5,
        label_smoothing=0.1,
        no_decay_keys=None,
        mixup_alpha=None,
        model_init="he_fout",
        validation_frequency=1,
        print_frequency=10,
        n_worker=8,
        resize_scale=0.08,
        distort_color="tf",
        image_size=224,
        robust_mode = False,
        epsilon = 0.031,
        num_steps = 10,
        step_size = 0.0078,
        clip_min = 0,
        clip_max = 1,
        const_init = False,
        beta = 6.0,
        distance = "l_inf",
        train_criterion = "trades",
        test_criterion = "ce",
        kd_criterion = 'rslad',
        attack_type = "linf-pgd",
        **kwargs
    ):
        super(DistributedClassificationRunConfig, self).__init__(
            n_epochs,
            init_lr,
            lr_schedule_type,
            lr_schedule_param,
            dataset,
            train_batch_size,
            test_batch_size,
            valid_size,
            opt_type,
            opt_param,
            weight_decay,
            label_smoothing,
            no_decay_keys,
            mixup_alpha,
            model_init,
            validation_frequency,
            print_frequency,
            n_worker,
            resize_scale,
            distort_color,
            image_size,
            robust_mode,
            epsilon,
            num_steps,
            step_size,
            clip_min,
            clip_max,
            const_init,
            beta,
            distance,
            epsilon,
            num_steps * 2,
            step_size,
            clip_min,clip_max,
            const_init,
            beta,
            distance,
            train_criterion,
            test_criterion,
            kd_criterion,
            attack_type,
            **kwargs
        )

        self._num_replicas = kwargs["num_replicas"]
        self._rank = kwargs["rank"]

    @property
    def data_provider(self):
        if self.__dict__.get("_data_provider", None) is None:
            if self.dataset == ImagenetDataProvider.name():
                DataProviderClass = ImagenetDataProvider
            elif self.dataset == Cifar10DataProvider.name():
                DataProviderClass = Cifar10DataProvider  
            elif self.dataset == Cifar100DataProvider.name():
                DataProviderClass = Cifar100DataProvider        
            else:
                raise NotImplementedError
            if self.dataset == "imagenet":
                self.__dict__["_data_provider"] = DataProviderClass(
                    train_batch_size=self.train_batch_size,
                    test_batch_size=self.test_batch_size,
                    valid_size=self.valid_size,
                    n_worker=self.n_worker,
                    resize_scale=self.resize_scale,
                    distort_color=self.distort_color,
                    image_size=self.image_size,
                    num_replicas=self._num_replicas,
                    rank=self._rank,
                )
            else:
                self.__dict__["_data_provider"] = DataProviderClass(
                    train_batch_size=self.train_batch_size,
                    test_batch_size=self.test_batch_size,
                    valid_size=self.valid_size,
                    n_worker=self.n_worker,
                    resize_scale=None,
                    distort_color=None,
                    image_size=self.image_size,
                    num_replicas=self._num_replicas,
                    rank=self._rank,   
                ) 
        return self.__dict__["_data_provider"]
    @property
    def train_criterion_loss (self):
        if self.train_criterion == "trades" : 
            return trades_loss
        elif self.train_criterion == "mart" : 
            return mart_loss 
        elif self.train_criterion == "sat"  :
            return sat_loss
        elif self.train_criterion == "hat" : 
            return hat_loss
    @property
    def test_criterion_loss (self) : 
        if self.test_criterion == "ce" : 
            return torch.nn.CrossEntropyLoss()     
    @property 
    def kd_criterion_loss (self) : 
        if self.kd_criterion =="ard" :
            return ard_loss
        elif self.kd_criterion == "adaad" : 
            return adaad_loss 
        elif self.kd_criterion == "rslad" : 
            return   rslad_loss 

        