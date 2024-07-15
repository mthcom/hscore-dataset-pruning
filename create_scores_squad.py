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
Create H-Score and ambiguous scores for SQuAD tasks.
"""

import torch, os, argparse
from pathlib import Path

def flatten_outputs(outputs, nb_epochs, nb_labels = 384):
    result_start = list()
    result_end = list()
    for i, o in enumerate(outputs):
        result_start+= o[0].flatten().tolist()
        result_end+= o[1].flatten().tolist()
    return torch.as_tensor([result_start, result_end]).view(2, nb_epochs, -1, nb_labels).permute(1,0,2,3)

def add_to_rewards(task_name, nb_epochs, create_hscore):
    rewards = None
    for l in os.listdir():
        if f"TASK:{task_name}" not in l:
            continue
        try:
            outputs = torch.load(os.path.join(l, 'all_outputs.bin'), map_location='cpu')
            outputs = flatten_outputs(outputs, nb_epochs, nb_labels=len(outputs[0][0][0]))
            sampler = torch.load(os.path.join(l, 'sampler.bin'), map_location='cpu')
            labels = torch.load(os.path.join(l, 'labels.bin'), map_location='cpu')
        except FileNotFoundError as e:
            print(e.filename, 'not found')
            continue
        else:
            print('processing ', l)
        if rewards == None:
            rewards = [[list() for j in range(labels.shape[1])] for i in range(nb_epochs)]
        
        outputs[:] = outputs[:,:,sampler.argsort(),:]

        if create_hscore:
            outputs = outputs.argmax(dim = 3)
            curr_reward = (labels - outputs).abs().sum(dim = 1)
            for epoch_idx in range(curr_reward.shape[0]):
                for sample_idx in range(curr_reward.shape[1]):
                    rewards[epoch_idx][sample_idx].append(curr_reward[epoch_idx][sample_idx])
        else:
            for epoch_idx in range(outputs.shape[0]):
                for sample_idx in range(outputs.shape[2]):
                    reward_start = torch.nn.functional.softmax(outputs[epoch_idx][0][sample_idx])[labels[0][sample_idx]]
                    reward_end = torch.nn.functional.softmax(outputs[epoch_idx][1][sample_idx])[labels[1][sample_idx]]
                    rewards[epoch_idx][sample_idx].append(reward_start + reward_end)

    rewards = torch.as_tensor(rewards)
    if create_hscore:
        rewards = rewards == 0
    return rewards

parser = argparse.ArgumentParser()
parser.add_argument('--models', type=str, required=True, help='Comma seperated list of models. Supported models are "roberta_large", "opt350"')
parser.add_argument('--outputs_path', type=str, required=True, help='Path of the folder containing all the outputs from previous runs')
parser.add_argument('--scores_folder', type=str, default="scores", help='Path of the folder to put the generated scores')
parser.add_argument('--nb_epochs', type=int, default=2, help='Number of epochs to use to create scores.')

args = parser.parse_args()
args.models = args.models.replace(' ', '').split(',')
task_name = 'squad_v2'

create_hscore = True
for create_hscore in [True, False]:
    method_name = "hscore" if create_hscore else "ambiguous"
    for m in args.models:
        print(method_name, m, task_name)

        folder_path = os.path.join(args.scores_folder, method_name, m)
        file_path = os.path.join(folder_path, task_name)

        if os.path.isfile(file_path):
            print(f"score file already exists at {file_path}, skipping...")
            continue

        current_dir = os.getcwd()
        try:
            os.chdir(os.path.join(args.outputs_path, m))
        except FileNotFoundError as e:
            print("Make sure directory", e.filename, "exists.")
            exit(1)
        rewards = add_to_rewards(task_name, args.nb_epochs, create_hscore)
        os.chdir(current_dir)

        Path(folder_path).mkdir(parents=True, exist_ok=True)
        print('saving', file_path)
        torch.save(rewards, file_path)
