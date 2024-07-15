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
Create H-Score and ambiguous scores for RACE task.
"""

import torch, os, argparse
from pathlib import Path
from datasets import load_dataset

def load_labels(model_name):
    dataset = load_dataset('race', 'all', cache_dir='scores_cache_dir')
    converter = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    labels = torch.as_tensor([converter[i] for i in dataset['train']['answer']])
    if 'opt' in model_name:
        new_labels = torch.zeros((labels.shape[0], labels.unique().shape[0]), dtype = int)
        new_labels[torch.arange(new_labels.shape[0]), labels] = 1
        labels = new_labels.flatten()
    return labels

def flatten_outputs(outputs, num_labels):
    result = list()
    for o in outputs:
        result += o.flatten().tolist()
    result = torch.as_tensor(result)
    result = result.view(args.nb_epochs, -1, num_labels)
    return result

def add_to_rewards():
    rewards = [[list() for j in range(labels.shape[0])] for i in range(args.nb_epochs)]
    for l in os.listdir():
        if f"TASK:{task_name}" not in l:
            continue
        try:
            outputs = torch.load(os.path.join(l, 'all_outputs.bin'), map_location='cpu')
            outputs = flatten_outputs(outputs, len(outputs[0][0]))
            sampler = torch.load(os.path.join(l, 'sampler.bin'), map_location='cpu')
        except FileNotFoundError as e:
            print(e.filename, 'not found')
            continue
        else:
            print('processing ', l)

        if create_hscore:
            outputs = outputs.argmax(dim = -1)
        for epoch_idx in range(args.nb_epochs):
            for idx, sampler_idx in enumerate(sampler):
                if idx < outputs.shape[1]:
                    if not create_hscore:
                        reward = torch.nn.functional.softmax(outputs[epoch_idx][idx])[labels[sampler_idx]]
                    else:
                        reward = int(outputs[epoch_idx][idx] == labels[sampler_idx])
                else:
                    reward = 0
                rewards[epoch_idx][sampler_idx].append(reward)
    rewards = torch.as_tensor(rewards)
    return rewards

parser = argparse.ArgumentParser(description='Argument Parser Example')
parser.add_argument('--models', type=str, required=True, help='Comma seperated list of models. Supported models are "roberta_large", "opt350"')
parser.add_argument('--outputs_path', type=str, required=True, help='Path of the folder containing all the outputs from previous runs')
parser.add_argument('--scores_folder', type=str, default="scores", help='Path of the folder to put the generated scores')
parser.add_argument('--nb_epochs', type=int, default=3, help='Number of epochs to use to create scores.')

args = parser.parse_args()
args.models = args.models.replace(' ', '').split(',')

task_name = 'race'
labels_by_model = {k : load_labels(k) for k in args.models}

for create_hscore in [True, False]:
    method_name = "hscore" if create_hscore else "ambiguous"
    for m in args.models:
        folder_path = os.path.join(args.scores_folder, method_name, m)
        file_path = os.path.join(folder_path, task_name)
        if os.path.isfile(file_path):
            print(f"score file already exists at {file_path}, skipping...")
            continue
    
        labels = labels_by_model[m][:-10]

        current_dir = os.getcwd()
        try:
            os.chdir(os.path.join(args.outputs_path, m))
        except FileNotFoundError as e:
            print("Make sure directory", e.filename, "exists.")
            exit(1)
        rewards = add_to_rewards()
        os.chdir(current_dir)

        Path(folder_path).mkdir(parents=True, exist_ok=True)
        print('saving', file_path)
        torch.save(rewards, file_path)
