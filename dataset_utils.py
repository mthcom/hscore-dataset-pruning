#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Sayed Mohammadreza Tayaranian Hosseini. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tools to apply various dataset pruning methods to the training dataset.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from types import MethodType
import torch, transformers, os

class PruningMethod(Enum):
    HSCORE='hscore'
    AMBIGUOUS='ambiguous'
    RANDOM='random'

@dataclass
class DataPruningArguments:
    """
    Arguments exclusively used for methods related to HScore and other dataset pruning methods.
    """

    pruning_method: str = field(
        default=None, metadata={"help": "Indicate the method used for pruning the dataset. Must be one of \'ambiguous\', \'hscore\', and \'random\'."}
    )
    retention_percentage: Optional[float] = field(
        default=None,
        metadata={
            "help": ( "Manually set the retention percentage for random pruning and cartography-based pruning."
            )
        },
    )
    scores_to_remove: Optional[str] = field(
        default='06',
        metadata={
            "help": ( "A string containing multiple digits. Each digit corresponds to an H-Score value that will be removed during fine-tuning."
            )
        },
    )
    sampler_seed: int = field(
        default=234,
        metadata={
            "help": ('Random seed to generate the sampler.'
            )
        },
    )
    load_sampler: str = field(
        default=None,
        metadata={
            "help": ('Path to the saved sampler torch tensor.'
            )
        },
    )
    rewards_file: str = field(
        default=None,
        metadata={
            "help": ('Path to the rewards file, used for dataset pruning with H-Score.'
            )
        },
    )
    label_probs_file: str = field(
        default=None,
        metadata={
            "help": ('Path to the file containing golden_labels, used for dataset pruning with ambiguous method.'
            )
        },
    )
    runs_used: int = field(
        default=6,
        metadata={
            "help": ('Number of runs used to compute the hscore.')  
        },
    )
    epochs_used: int = field(
        default=3,
        metadata={
            "help": ('Number of epochs of each run used to compute the hscore.')  
        },
    )

    def __post_init__(self):
        self._h_scores = None
        if self.epochs_used < 1 or self.epochs_used > 3:
            raise Exception("epochs_used is an integer between 1 and 3, inclusive.")
        self.scores_to_remove = [int(i) for i in self.scores_to_remove]
        if self.pruning_method is not None:
            self.pruning_method = PruningMethod(self.pruning_method)

    @property
    def h_scores(self):
        if self._h_scores is None:
            if self.rewards_file is None:
                raise Exception("rewards_file needs to specified in order to obtain h_scores.")
            rewards = torch.load(self.rewards_file)
            self._h_scores = rewards[:self.epochs_used].prod(dim = 0)[:,:self.runs_used].sum(dim = 1)
        return self._h_scores

def prune_train_dataset(data_sampler : list, datapruning_args : DataPruningArguments, trainer : transformers.trainer, logger):
    if datapruning_args.pruning_method is not None:
        if datapruning_args.pruning_method in [PruningMethod.AMBIGUOUS, PruningMethod.RANDOM]:
            if datapruning_args.retention_percentage is not None:
                retention_percentage = datapruning_args.retention_percentage
                nb_retention_samples = int(len(data_sampler) * retention_percentage)
            else:
                nb_retention_samples = int(datapruning_args.h_scores.numel() - sum([(datapruning_args.h_scores == s).sum() for s in datapruning_args.scores_to_remove]))
            if datapruning_args.pruning_method == PruningMethod.AMBIGUOUS:
                golden_label_probs = torch.load(datapruning_args.label_probs_file)[:,:,:datapruning_args.runs_used].permute(1,0,2).reshape(-1, datapruning_args.epochs_used * datapruning_args.runs_used)
                confidence = golden_label_probs.mean(dim = 1)
                variability = golden_label_probs.std(dim = 1)
                pruning_metric = variability
                descending = True
                
                good_samples = pruning_metric.argsort(descending=descending)[:nb_retention_samples]
                data_sampler_tmp = list()
                for ds_idx, ds in enumerate(data_sampler):
                    if ds in good_samples:
                        data_sampler_tmp.append(ds)
            elif datapruning_args.pruning_method is PruningMethod.RANDOM:
                data_sampler_tmp = data_sampler[:nb_retention_samples]
        elif datapruning_args.pruning_method == PruningMethod.HSCORE:
            data_sampler_tmp = list()
            for ds in data_sampler:
                if datapruning_args.h_scores[ds] not in datapruning_args.scores_to_remove:
                    data_sampler_tmp.append(ds)
        
        pruning_log_str = ''
        pruning_log_str += f'pruning method:{datapruning_args.pruning_method.value}\n'
        pruning_log_str += f'original dataset size:{len(data_sampler)}\n'
        pruning_log_str += f'pruned dataset size:{len(data_sampler_tmp)}\n'
        pruning_log_str += f'retention percentage:{100 * (len(data_sampler_tmp) / len(data_sampler)):.2f}\n'
        logger.info(f"Pruning info:{pruning_log_str}")
        with open(os.path.join(trainer.args.output_dir, 'pruning_stat'), 'w') as f:
            f.write(pruning_log_str)
    
        data_sampler = data_sampler_tmp
    
    def _get_train_sampler(self):
        return data_sampler
    if data_sampler is not None:
        trainer._get_train_sampler = MethodType(_get_train_sampler, trainer)
