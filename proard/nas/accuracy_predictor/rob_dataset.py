# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data

from proard.utils import list_mean

__all__ = ["net_setting2id", "net_id2setting", "RobustnessDataset"]


def net_setting2id(net_setting):
    return json.dumps(net_setting)


def net_id2setting(net_id):
    return json.loads(net_id)


class RegDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        super(RegDataset, self).__init__()
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return self.inputs.size(0)


class RobustnessDataset:
    def __init__(self, path):
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    @property
    def net_id_path(self):
        return os.path.join(self.path, "net_id.dict")

    @property
    def rob_src_folder(self):
        return os.path.join(self.path, "src_rob")
    @property
    def rob_dict_path(self):
        return os.path.join(self.path, "src_rob/rob.dict")

    # TODO: support parallel building
    def build_rob_dataset(
        self, run_manager, dyn_network, n_arch=2000, image_size_list=None
    ):
        # load net_id_list, random sample if not exist
        if os.path.isfile(self.net_id_path):
            net_id_list = json.load(open(self.net_id_path))
        else:
            net_id_list = set()
            while len(net_id_list) < n_arch:
                net_setting = dyn_network.sample_active_subnet()
                net_id = net_setting2id(net_setting)
                net_id_list.add(net_id)
            net_id_list = list(net_id_list)
            net_id_list.sort()
            json.dump(net_id_list, open(self.net_id_path, "w"), indent=4)

        image_size_list = (
            [128, 160, 192, 224] if image_size_list is None else image_size_list
        )

        with tqdm(
            total=len(net_id_list) * len(image_size_list), desc="Building Robustness Dataset"
        ) as t:
            for image_size in image_size_list:
                # load val dataset into memory
                val_dataset = []
                run_manager.run_config.data_provider.assign_active_img_size(image_size)
                for images, labels in run_manager.run_config.valid_loader:
                    val_dataset.append((images, labels))
                # save path
                os.makedirs(self.rob_src_folder, exist_ok=True)
            
                rob_save_path = os.path.join(
                    self.rob_src_folder, "%d.dict" % image_size
                )
    
                rob_dict ={}
                # load existing rob dict  
                if os.path.isfile(rob_save_path):    
                    existing_rob_dict = json.load(open(rob_save_path,"r"))
                else:
                    existing_rob_dict = {}
                for net_id in net_id_list:
                    net_setting = net_id2setting(net_id)
                    key = net_setting2id({**net_setting, "image_size": image_size})
                    if key in existing_rob_dict:
                        rob_dict[key] = existing_rob_dict[key]
                        t.set_postfix(
                            {
                                "net_id": net_id,
                                "image_size": image_size,
                                "info_rob" : rob_dict[key],
                                "status": "loading",
                            }
                        )
                        t.update()
                        continue
                    dyn_network.set_active_subnet(**net_setting)
                    run_manager.reset_running_statistics(dyn_network)
                    net_setting_str = ",".join(
                        [
                            "%s_%s"
                            % (
                                key,
                                "%.1f" % list_mean(val)
                                if isinstance(val, list)
                                else val,
                            )
                            for key, val in net_setting.items()
                        ]
                    )
                    loss, (top1, top5,robust1,robust5) = run_manager.validate(
                        run_str=net_setting_str,
                        net=dyn_network,
                        data_loader=val_dataset,
                        no_logs=True,
                    )
                    info_robust = robust1
                    t.set_postfix(
                        {
                            "net_id": net_id,
                            "image_size": image_size,
                            "info_rob" : info_robust,
                            "info_robust" : info_robust,
                        }
                    )
                    t.update()

                    rob_dict.update({key:info_robust})
                    json.dump(rob_dict, open(rob_save_path, "w"), indent=4)

    def merge_rob_dataset(self, image_size_list=None):
        # load existing data
        merged_rob_dict = {}
        for fname in os.listdir(self.rob_src_folder):
            if ".dict" not in fname:
                continue
            image_size = int(fname.split(".dict")[0])
            if image_size_list is not None and image_size not in image_size_list:
                print("Skip ", fname)
                continue
            full_path = os.path.join(self.rob_src_folder, fname)
            partial_rob_dict = json.load(open(full_path))
            merged_rob_dict.update(partial_rob_dict)
            print("loaded %s" % full_path)
        json.dump(merged_rob_dict, open(self.rob_dict_path, "w"), indent=4)
        return merged_rob_dict

    def build_rob_data_loader(
        self, arch_encoder, n_training_sample=None, batch_size=256, n_workers=16
    ):
        # load data
        rob_dict = json.load(open(self.rob_dict_path))
        X_all_rob = []
        Y_all_rob = []
        with tqdm(total=len(rob_dict), desc="Loading data") as t:
            for k, v in rob_dict.items():
                dic = json.loads(k)
                X_all_rob.append(arch_encoder.arch2feature(dic))
                Y_all_rob.append(v / 100.0)  # range: 0 - 1
                t.update()        
        base_rob = np.mean(Y_all_rob)
        # convert to torch tensor
        X_all_rob = torch.tensor(X_all_rob, dtype=torch.float)
        Y_all_rob = torch.tensor(Y_all_rob)

        # random shuffle
        shuffle_idx_rob = torch.randperm(len(X_all_rob))
        X_all_rob = X_all_rob[shuffle_idx_rob]
        Y_all_rob = Y_all_rob[shuffle_idx_rob]
        # split data
        idx_rob = X_all_rob.size(0) // 5 * 4 if n_training_sample is None else n_training_sample
        val_idx_rob = X_all_rob.size(0) // 5 * 4
        X_train_rob, Y_train_rob = X_all_rob[:idx_rob], Y_all_rob[:idx_rob]
        X_test_rob, Y_test_rob = X_all_rob[val_idx_rob:], Y_all_rob[val_idx_rob:]
        print("Train Robustness Size: %d," % len(X_train_rob), "Valid Robustness Size: %d" % len(X_test_rob))
        # build data loader
        train_dataset_rob = RegDataset(X_train_rob, Y_train_rob)
        val_dataset_rob = RegDataset(X_test_rob, Y_test_rob)
    
        train_loader_rob = torch.utils.data.DataLoader(
            train_dataset_rob,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=n_workers,
        )
        valid_loader_rob = torch.utils.data.DataLoader(
            val_dataset_rob,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=n_workers,
        )
        return train_loader_rob, valid_loader_rob , base_rob


