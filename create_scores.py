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
Create H-Score and ambiguous scores for GLUE tasks + SNLI.
"""

import torch, os, argparse
from pathlib import Path
from datasets import load_dataset

def load_labels(task_name):
    if task_name in ['mnli', 'sst2']:
        dataset = load_dataset('glue', task_name, cache_dir='scores_cache_dir')
        labels = dataset['train']['label']
        labels = torch.as_tensor(labels)
    elif task_name == 'snli':
        dataset = load_dataset(task_name, cache_dir='scores_cache_dir')
        labels = dataset['train']['label']
        labels = torch.as_tensor(labels)
        labels = labels[labels != -1]
    return labels

def flatten_outputs(outputs, nb_labels, nb_epochs):
    result = list()
    for i, o in enumerate(outputs):
        iterator = o.logits if hasattr(o, 'logits') else o
        for l in iterator:
            result.append(l.tolist())
    return torch.as_tensor(result).view(nb_epochs, -1, nb_labels)

def add_to_rewards(task_name, labels, nb_epochs, create_hscore):
    rewards = [[list() for j in range(len(labels))] for i in range(nb_epochs)]
    for l in os.listdir():
        if f"TASK:{task_name}" not in l:
            continue
        try:
            outputs = flatten_outputs(torch.load(os.path.join(l, 'all_outputs.bin'), map_location='cpu'), nb_labels=max(labels) + 1, nb_epochs=nb_epochs)
            sampler = torch.load(os.path.join(l, 'sampler.bin'))
        except FileNotFoundError as e:
            print(e.filename, 'not found')
            continue
        else:
            print('processing ', l)
        
        for epoch_idx in range(nb_epochs):
            for idx, sampler_idx in enumerate(sampler):
                curr_outputs = outputs[epoch_idx][idx]
                if not create_hscore:
                    label_idx = labels[sampler_idx]
                    reward = torch.nn.functional.softmax(curr_outputs)[label_idx]
                else:
                    reward = int(curr_outputs.argmax() == labels[sampler_idx])
                rewards[epoch_idx][sampler_idx].append(reward)
    return torch.as_tensor(rewards)

parser = argparse.ArgumentParser()
parser.add_argument('--tasks', type=str, required=True, help='Comma seperated list of tasks. Supported tasks are "mnli", "snli", "sst2"')
parser.add_argument('--models', type=str, required=True, help='Comma seperated list of models. Supported models are "roberta_large", "opt350"')
parser.add_argument('--outputs_path', type=str, required=True, help='Path of the folder containing all the outputs from previous runs')
parser.add_argument('--scores_folder', type=str, default="scores", help='Path of the folder to put the generated scores')
parser.add_argument('--nb_epochs', type=int, default=3, help='Number of epochs to use to create scores.')

args = parser.parse_args()
args.tasks = args.tasks.replace(' ', '').split(',')
args.models = args.models.replace(' ', '').split(',')

labels_by_task = {k : load_labels(k) for k in args.tasks}

create_hscore = True
for create_hscore in [True, False]:
    method_name = "hscore" if create_hscore else "ambiguous"
    for m in args.models:
        for task_name in args.tasks:
            print(method_name, m, task_name)

            folder_path = os.path.join(args.scores_folder, method_name, m)
            file_path = os.path.join(folder_path, task_name)
            if os.path.isfile(file_path):
                print(f"score file already exists at {file_path}, skipping...")
                continue
            
            labels = labels_by_task[task_name]

            current_dir = os.getcwd()
            try:
                os.chdir(os.path.join(args.outputs_path, m))
            except FileNotFoundError as e:
                print("Make sure directory", e.filename, "exists.")
                exit(1)
            rewards = add_to_rewards(task_name, labels, args.nb_epochs, create_hscore)
            os.chdir(current_dir)

            Path(folder_path).mkdir(parents=True, exist_ok=True)
            print('saving', file_path)
            torch.save(rewards, file_path)
