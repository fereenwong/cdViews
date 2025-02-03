import json
import torch
import yaml
from easydict import EasyDict


def convert_value(value):
    if value.lower() == 'none':
        return None
    elif value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    return value


def update_args_from_cfg(args, cfg, parent_key=''):
    for key, value in cfg.items():
        # Construct the full key path
        full_key = f"{parent_key}.{key}" if parent_key else key

        if isinstance(value, str):
            value = convert_value(value)

        if isinstance(value, dict):
            # Recursively update args for nested dictionaries
            update_args_from_cfg(args, value, full_key)
        else:
            # Update args with the current key-value pair
            if full_key in vars(args):
                setattr(args, full_key, value)
            else:
                # Handle nested keys in argparse
                keys = full_key.split('.')
                sub_args = args
                for k in keys[:-1]:
                    if not hasattr(sub_args, k):
                        setattr(sub_args, k, EasyDict())
                    sub_args = getattr(sub_args, k)
                setattr(sub_args, keys[-1], value)
    return args


def load_and_update(args):
    with open(args.cfg_file, 'r') as f:
        try:
            config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.safe_load(f)

    update_args_from_cfg(args, config)
    return args


def get_scanqa(cfg, mode=None):
    if mode is None:
        scanqa_train = json.load(open(cfg.ScanQA_file.train))
        scanqa_val = json.load(open(cfg.ScanQA_file.val))
        scanqa_test_w_obj = json.load(open(cfg.ScanQA_file.test_w_obj))
        scanqa_test_wo_obj = json.load(open(cfg.ScanQA_file.test_wo_obj))
        return scanqa_train, scanqa_val, scanqa_test_w_obj, scanqa_test_wo_obj
    else:
        mode_to_file = {
            "train": cfg.ScanQA_file.train,
            "val": cfg.ScanQA_file.val,
            "test_w_obj": cfg.ScanQA_file.test_w_obj,
            "test_wo_obj": cfg.ScanQA_file.test_wo_obj
        }
        if mode in mode_to_file:
            scanqa = json.load(open(mode_to_file[mode]))
            return scanqa
        else:
            raise ValueError(f"Invalid mode '{mode}'. Choose from {list(mode_to_file.keys())}.")


def get_sqa(cfg, mode=None):
    if mode is None:
        sqa_train = json.load(open(cfg.SQA_file.train))
        sqa_val = json.load(open(cfg.SQA_file.val))
        sqa_test = json.load(open(cfg.SQA_file.test))
        return sqa_train, sqa_val, sqa_test
    else:
        mode_to_file = {
            "train": cfg.SQA_file.train,
            "val": cfg.SQA_file.val,
            "test": cfg.SQA_file.test
        }
        if mode in mode_to_file:
            sqa = json.load(open(mode_to_file[mode]))
            return sqa
        else:
            raise ValueError(f"Invalid mode '{mode}'. Choose from {list(mode_to_file.keys())}.")


def custom_collate_fn(batch):
    qs_id_list, qs_list, feat_list, label_list, image_file_list = [], [], [], [], []
    for (qs_id, qs, feat, label, image_files) in batch:
        qs_id_list.append(qs_id)
        qs_list.append(qs)
        feat_list.append(feat)
        label_list.append(label)
        image_file_list.append(image_files)

    if label_list[-1] is None:
        label_list = None
    else:
        label_list = torch.cat(label_list)
    return qs_id_list, qs_list, torch.cat(feat_list), label_list, image_file_list
