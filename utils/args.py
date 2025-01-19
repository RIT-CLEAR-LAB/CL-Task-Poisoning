# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')

    parser.add_argument('--n_epochs', type=int,
                        help='Batch size.')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size.')

    parser.add_argument('--distributed', type=str, default='no', choices=['no', 'dp', 'ddp'])

    parser.add_argument('--n_image_poisonings', default=None, type=int,
                        help='number of image noise poisonings created when creating evenly spaced poisonings.')
    parser.add_argument("--image_poisoning_type", default=None, choices=[-1, 0, 1, 2, 3, 4, 5], type=int,
                        help="Choose poisoning transform to be applied to training data with n_image_poisonings: \
                        Defocus Blur -> 0, Gaussian Noise -> 1, Shot Noise -> 2, Speckle Noise -> 3, \
                        Pixel Permutation -> 4, Identity (No transform) -> 5 \
                        If set to -1, no poisoning will be applied.")

    parser.add_argument('--n_label_flip_poisonings', default=None, type=int,
                        help='number of past label flip poisonings created when creating evenly spaced poisonings.')
    parser.add_argument('--label_flip_percentage', default=None, choices=list(range(101)), type=int, 
                        help='Choose the percentage (0 ~ 100) of poisoned samples to be included \
                        in training batch with n_label_flip_poisonings.')
    
    parser.add_argument('--n_backdoor_poisonings', default=None, type=int,
                        help='number of backdoor poisonings created when creating evenly spaced poisonings.')
    parser.add_argument('--poisoning_rate', default=None, choices=list(range(101)), type=int, 
                        help='Choose the percentage (0 ~ 100) of data to be backdoor poisoned \
                        in training batch with n_backdoor_poisonings.')
    parser.add_argument('--trigger_rate', default=None, choices=list(range(101)), type=int, 
                        help='Choose the percentage (0 ~ 100) of data to be backdoor tagged \
                        in test batch with n_backdoor_poisonings.')

    parser.add_argument('--poisoning_severity', default=1, choices=[1, 2, 3, 4, 5], type=int,
                        help='Choose the intensity of the poisoning transform (1 ~ 5).')
    parser.add_argument('--classes_per_poisoning', type=int, default=0,
                        help='Number of classes that can be poisoned at once. \
                        If set to 0 (default), all previous classes will be poisoned.')

def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--non_verbose', default=0, choices=[0, 1], type=int, help='Make progress bars non verbose')
    parser.add_argument('--disable_log', default=0, choices=[0, 1], type=int, help='Enable csv logging')

    parser.add_argument('--validation', default=0, choices=[0, 1], type=int,
                        help='Test on the validation set')
    parser.add_argument('--ignore_other_metrics', default=0, choices=[0, 1], type=int,
                        help='disable additional metrics')
    parser.add_argument('--debug_mode', type=int, default=0, help='Run only a few forward steps per epoch')
    parser.add_argument('--nowand', default=0, choices=[0, 1], type=int, help='Inhibit wandb logging')
    parser.add_argument('--wandb_entity', type=str, default='regaz', help='Wandb entity')
    parser.add_argument('--wandb_project', type=str, default='mammoth', help='Wandb project name')


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int,
                        help='The batch size of the memory buffer.')
    parser.add_argument('--buffer_mode', default='balanced', type=str,
                        choices=['ring', 'reservoir', 'balanced'], 
                        help='The method for buffer sampling.')
    parser.add_argument('--buffer_retrieve_mode', default='random', type=str,
                        choices=['random', 'mir', 'min_rehearsal', 'min_margin',
                                'min_logit_distance', 'min_confidence', 'max_loss'], 
                        help='The method for buffer updating.')
