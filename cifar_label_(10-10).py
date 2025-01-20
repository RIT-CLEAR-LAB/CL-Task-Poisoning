import itertools
import argparse
import numpy  # needed (don't change it)
import importlib
import os
import socket
import sys
import setproctitle
import torch

import datetime
import uuid
from argparse import ArgumentParser


mammoth_path = '/home/jw7630/repos/CL-Task-Poisoning' #os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # nopep8
sys.path.append(mammoth_path)  # nopep8
sys.path.append(mammoth_path + '/datasets')  # nopep8
sys.path.append(mammoth_path + '/backbone')  # nopep8
sys.path.append(mammoth_path + '/models')  # nopep8


from datasets import NAMES as DATASET_NAMES  # nopep8
from datasets import ContinualDataset, get_dataset  # nopep8
from models import get_all_models, get_model  # nopep8

from utils.args import add_management_args
from utils.best_args import best_args
from utils.conf import set_random_seed
from utils.continual_training import train as ctrain
from utils.distributed import make_dp
from utils.training import train


def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib  # pyright: ignore
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

def main(args=None, log_filename=None, device="cuda"):
    os.putenv("MKL_SERVICE_FORCE_INTEL", "1")
    os.putenv("NPY_MKL_FORCE_INTEL", "1")

    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()
    dataset = get_dataset(args)

    if args.batch_size is None:
        args.batch_size = dataset.get_batch_size()
    if hasattr(importlib.import_module('models.' + args.model), 'Buffer') and args.minibatch_size is None:
        args.minibatch_size = dataset.get_minibatch_size()

    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())
    model.device = torch.device(device)
    model.buffer.device = torch.device(device)

    # set job name
    setproctitle.setproctitle('{}_{}_{}'.format(args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))

    if isinstance(dataset, ContinualDataset):
        train(model, dataset, args, log_filename)
    else:
        assert not hasattr(model, 'end_task') or model.NAME == 'joint_gcl'
        ctrain(args)

models = ["er_ace"] #["xder", "er", "der", "derpp", "er_ace"]
buffer_sizes = [200] #[200, 500, 2000]
label_flip_percentages = [0, 25, 50, 75]

for buffer_size, model, label_flip_percentage in itertools.product(buffer_sizes, models, label_flip_percentages):
    args_dict = dict(
        dataset='seq-cifar100-label-poisoning1010',
        model=model,
        n_epochs=50,
        nowand=1,
        non_verbose=1,
        ignore_other_metrics=1,
        seed=48, #45, #42,
        n_label_flip_poisonings=1,
        poisoning_severity=5,
        classes_per_poisoning=0,
        buffer_mode="reservoir",
        n_image_poisonings=None, # fails without this
        debug_mode=0, # fails without this
        disable_log=0, # fails without this
        batch_size=128, # not in best args
        minibatch_size=128, # not in best args
        buffer_size=buffer_size,
        label_flip_percentage=label_flip_percentage
    )

    if buffer_size == 200:
        bs_config = 500
    elif buffer_size == 5000:
        bs_config = 2000
    else:
        bs_config = buffer_size

    optim_params = best_args['seq-cifar100'][model][bs_config]
    args_dict.update(optim_params)
    args = argparse.Namespace(**args_dict)
    file = f"results/label_poison/cifar_{model}_{buffer_size}_{label_flip_percentage}_(10-10)_2.json"
    main(args, file)