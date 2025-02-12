# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import sklearn.metrics
import sys
from argparse import Namespace
from typing import Tuple
import json
from datetime import datetime
import torch
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from utils.loggers import *
from utils.status import ProgressBar

try:
    import wandb
except ImportError:
    wandb = None


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
            dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')

def get_unique_labels(test_loader):
    labels = []
    for _, batch_labels, _ in test_loader:
        labels.extend(batch_labels.numpy())

    unique_labels = torch.unique(torch.tensor(labels))
    return unique_labels.tolist()

def calculate_backdoor_success(model: ContinualModel, dataset: ContinualDataset) -> Tuple[int, int]:
    status = model.net.training
    model.net.eval()

    predictions = []
    all_labels = []
    all_poisoned_flags = []

    for k, test_loader in enumerate(dataset.test_loaders):
        for data in test_loader:
            with torch.no_grad():
                inputs, labels, poisoned_flags = data[0], data[1], data[2]
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                poisoned_flags = poisoned_flags.to(model.device)

                outputs = model(inputs) if 'class-il' in model.COMPATIBILITY else model(inputs, k)
                _, pred = torch.max(outputs.data, 1)

                predictions.append(pred)
                all_labels.append(labels)
                all_poisoned_flags.append(poisoned_flags)

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)

    predictions = torch.cat(predictions, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
    all_poisoned_flags = torch.cat(all_poisoned_flags, dim=0).cpu().numpy()

    poisoned_indices = all_poisoned_flags == 1
    total_tagged = poisoned_indices.sum()
    successful_triggers = ((predictions[poisoned_indices] == 0) & (all_labels[poisoned_indices] != 0)).sum()

    model.net.train(status)

    return int(successful_triggers), int(total_tagged)


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        predictions = list()
        all_labels = list()

        # unique_labels = get_unique_labels(test_loader)
        # print("Unique labels in the test loader:", unique_labels)

        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data[0], data[1]
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)

                _, pred = torch.max(outputs.data, 1)

                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                predictions.append(pred)
                all_labels.append(labels)

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
        predictions = torch.cat(predictions, dim=0).cpu().numpy()
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

    model.net.train(status)
    return accs, accs_mask_classes


def train(model: ContinualModel, dataset: ContinualDataset, args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    print(args)

    if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        args.wandb_url = wandb.run.get_url()

    model.net.to(model.device)
    results, results_mask_classes = [], []
    buffer_contamination = []
    backdoor_success_rate = []

    if not args.disable_log:
        logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)

    progress_bar = ProgressBar(verbose=not args.non_verbose)

    if not args.ignore_other_metrics:
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            if (
                args.n_image_poisonings is None
                and args.n_label_flip_poisonings is None
                and args.n_backdoor_poisonings is None
            ):
                _, _ = dataset_copy.get_data_loaders()
            else:
                _, _ = dataset_copy.get_poisoned_data_loaders()
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            random_results_class, random_results_task = evaluate(model, dataset_copy)

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        model.net.train()
        if (
            args.n_image_poisonings is None
            and args.n_label_flip_poisonings is None
            and args.n_backdoor_poisonings is None
        ):
            train_loader, _ = dataset.get_data_loaders()
        else:
            train_loader, _ = dataset.get_poisoned_data_loaders()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t and not args.ignore_other_metrics:
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]

        scheduler = dataset.get_scheduler(model, args)
        for epoch in range(model.args.n_epochs):
            if args.model == 'joint':
                continue
            for i, data in enumerate(train_loader):
                if args.debug_mode and i > 3:
                    break

                inputs, labels, not_aug_inputs, poisoned_flags = data[0], data[1], data[2], data[3]
                inputs = inputs.to(model.device)
                labels = labels.to(model.device)
                not_aug_inputs = not_aug_inputs.to(model.device)

                if hasattr(dataset.train_loader.dataset, 'logits'):
                    logits = data[-1]
                    logits = logits.to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs, poisoned_flags, logits)
                else:
                    loss = model.meta_observe(inputs, labels, not_aug_inputs, poisoned_flags)

                poisoned_buffer_samples = model.check_buffer_contamination()
                buffer_contamination.append(poisoned_buffer_samples)
                assert not math.isnan(loss)
                progress_bar.prog(i, len(train_loader), epoch, t, loss)

            if scheduler is not None:
                scheduler.step()

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        metrics = evaluate(model, dataset)
        if 'backdoor' in dataset.NAME:
            successful_triggers, total_tagged = calculate_backdoor_success(model, dataset)
            backdoor_success_rate.append((successful_triggers, total_tagged))
        accs = metrics[:2]
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        if not args.disable_log:
            logger.log(mean_acc)
            logger.log_fullacc(accs)

        if not args.nowand:
            d2 = {'RESULT_class_mean_accs': mean_acc[0], 'RESULT_task_mean_accs': mean_acc[1],
                  **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs[0])},
                  **{f'RESULT_task_acc_{i}': a for i, a in enumerate(accs[1])}}

            wandb.log(d2)

    log_filename = (
        f"../results/Task-Poisoning/{datetime.now().strftime('%m-%d-%y-%H-%M-%S')}-{args.dataset}-{args.model}-buf-{args.buffer_size}-severity-{args.poisoning_severity}-ret_mode-{args.buffer_retrieve_mode}"
        f"{'-poisoning-type-' + str(args.image_poisoning_type) if args.n_image_poisonings is not None else '-poisoning-percentage-' + str(args.label_flip_percentage) if args.n_label_flip_poisonings is not None else '-p-rate-' + str(args.poisoning_rate) + '-t-rate' + str(args.trigger_rate) if args.n_backdoor_poisonings is not None else '-no-poisoning'}.json"
    )

    with open(log_filename, "w") as jsonfile:
        json.dump(
            {
                "cil_accuracies": results,
                "til_accuracies": results_mask_classes,
                "buffer_contamination": buffer_contamination,
                "backdoor_success_rate": backdoor_success_rate,
            },
            jsonfile,
        )

    if not args.disable_log and not args.ignore_other_metrics:
        logger.add_bwt(results, results_mask_classes)
        logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            logger.add_fwt(results, random_results_class,
                           results_mask_classes, random_results_task)

    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if not args.nowand:
        wandb.finish()
