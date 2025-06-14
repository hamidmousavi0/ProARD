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

__all__ = ["net_setting2id", "net_id2setting", "AccuracyRobustnessDataset"]


def net_setting2id(net_setting):
    return json.dumps(net_setting)


def net_id2setting(net_id):
    return json.loads(net_id)


class TwoRegDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets_acc , targets_rob ):
        super(TwoRegDataset, self).__init__()
        self.inputs = inputs
        self.targets_acc = targets_acc
        self.targets_rob = targets_rob

    def __getitem__(self, index):
        return self.inputs[index], self.targets_acc[index] , self.targets_rob[index]

    def __len__(self):
        return self.inputs.size(0)


class AccuracyRobustnessDataset:
    def __init__(self, path):
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    @property
    def net_id_path(self):
        return os.path.join(self.path, "net_id.dict")

    @property
    def acc_rob_src_folder(self):
        return os.path.join(self.path, "src")
    @property
    def acc_rob_dict_path(self):
        return os.path.join(self.path, "src/acc_robust.dict")
    

    # TODO: support parallel building
    def build_acc_rob_dataset(
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
        print(image_size_list)
        with tqdm(
            total=len(net_id_list) * len(image_size_list), desc="Building Acc Dataset"
        ) as t:
            for image_size in image_size_list:
                # load val dataset into memory
                val_dataset = []
                run_manager.run_config.data_provider.assign_active_img_size(image_size)
                for images, labels in run_manager.run_config.valid_loader:
                    val_dataset.append((images, labels))
                # save path
                os.makedirs(self.acc_rob_src_folder, exist_ok=True)
                acc_rob_save_path = os.path.join(
                    self.acc_rob_src_folder, "%d.dict" % image_size
                )
                acc_rob_dict = {}
                # load existing acc dict
                if os.path.isfile(acc_rob_save_path):
                    existing_acc_rob_dict = json.load(open(acc_rob_save_path, "r"))
                else:
                    existing_acc_rob_dict = {}    
                for net_id in net_id_list:
                    net_setting = net_id2setting(net_id)
                    key = net_setting2id({**net_setting, "image_size": image_size})
                    if key in existing_acc_rob_dict:
                        acc_rob_dict[key] = existing_acc_rob_dict[key]
                        t.set_postfix(
                            {
                                "net_id": net_id,
                                "image_size": image_size,
                                "info_val": acc_rob_dict[key],
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
                    info_val = [top1,robust1]
                    t.set_postfix(
                        {
                            "net_id": net_id,
                            "image_size": image_size,
                            "info_val": info_val,
                        }
                    )
                    t.update()

                    acc_rob_dict.update({key: info_val})
                    json.dump(acc_rob_dict, open(acc_rob_save_path, "w"), indent=4)


    def merge_acc_dataset(self, image_size_list=None):
        # load existing data
        merged_acc_rob_dict = {}
        for fname in os.listdir(self.acc_rob_src_folder):
            if ".dict" not in fname:
                continue
            image_size = int(fname.split(".dict")[0])
            if image_size_list is not None and image_size not in image_size_list:
                print("Skip ", fname)
                continue
            full_path = os.path.join(self.acc_rob_src_folder, fname)
            partial_acc_rob_dict = json.load(open(full_path))
            merged_acc_rob_dict.update(partial_acc_rob_dict)
            print("loaded %s" % full_path)
        json.dump(merged_acc_rob_dict, open(self.acc_rob_dict_path, "w"), indent=4)
        return merged_acc_rob_dict

    def build_acc_data_loader(
        self, arch_encoder, n_training_sample=None, batch_size=256, n_workers=16
    ):
        # load data
        acc_rob_dict = json.load(open(self.acc_rob_dict_path))
        X_all = []
        Y_acc_all = []
        Y_rob_all = []

        with tqdm(total=len(acc_rob_dict), desc="Loading data") as t:
            for k, v in acc_rob_dict.items():
                dic = json.loads(k)
                X_all.append(arch_encoder.arch2feature(dic))
                Y_acc_all.append(v[0] / 100.0)  # range: 0 - 1
                Y_rob_all.append(v[1] / 100.0)
                t.update()       
        base_acc = np.mean(Y_acc_all)
        base_rob = np.mean(Y_rob_all)
        # convert to torch tensor
        X_all = torch.tensor(X_all, dtype=torch.float)
        Y_acc_all = torch.tensor(Y_acc_all)
        Y_rob_all = torch.tensor(Y_rob_all)
 

        # random shuffle
        shuffle_idx = torch.randperm(len(X_all))
        X_all = X_all[shuffle_idx]
        Y_acc_all = Y_acc_all[shuffle_idx]
        Y_rob_all = Y_rob_all[shuffle_idx]
        # split data
        idx = X_all.size(0) // 5 * 4 if n_training_sample is None else n_training_sample
        val_idx = X_all.size(0) // 5 * 4
        X_train, Y_acc_train, Y_rob_train = X_all[:idx], Y_acc_all[:idx], Y_rob_all[:idx]
        X_test, Y_acc_test , Y_rob_test = X_all[val_idx:], Y_acc_all[val_idx:] , Y_rob_all[val_idx:]
        print("Train Size: %d," % len(X_train), "Valid Size: %d" % len(X_test))

        # build data loader
        train_dataset = TwoRegDataset(X_train, Y_acc_train , Y_rob_train)
        val_dataset = TwoRegDataset(X_test, Y_acc_test ,Y_rob_test )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=n_workers,
        )
        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=n_workers,
        )
    
        return train_loader, valid_loader, base_acc, base_rob


