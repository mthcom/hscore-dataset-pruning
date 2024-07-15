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
Extending the Trainer class from the Huggingface Transformers library to add the functionality to save training logits to disk.
"""

import transformers
import torch
import os
from typing import Any, Dict, Union
from transformers.utils import is_apex_available
import json
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerControl, TrainerState

if is_apex_available():
    from apex import amp

class EvalCallback(transformers.TrainerCallback):
    def __init__(self) -> None:
        super().__init__()
        self.eval_metrics = []

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        self.eval_metrics.append(kwargs['metrics'])

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        with open(os.path.join(args.output_dir, 'eval_by_epoch.json'), 'w') as f:
            json.dump(self.eval_metrics, f)

class CustomTrainer(transformers.Trainer):
    """
    Extend Trainer and save training logits after training is over.

    Uses the same constructor as the parent class.
    Overrides training_step to extract logits.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_outputs = list()
        self.add_callback(EvalCallback())
        
    def train(self, *args, **kwargs):
        output = super().train(*args, **kwargs)
        torch.save(self.training_outputs, os.path.join(self.args.output_dir, 'all_outputs.bin')) # save training_outputs to disk
        return output
        
    def add_to_training_outputs(self, step_outputs):
        if 'logits' in step_outputs:
            self.training_outputs.append(step_outputs['logits'].detach().to('cpu'))
        elif 'start_logits' in step_outputs and 'end_logits' in step_outputs:
            self.training_outputs.append([step_outputs['start_logits'].detach().to('cpu'), step_outputs['end_logits'].detach().to('cpu')])
            
    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs and add the outputs to the list of training outputs.

        Overridden from the parent class.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs = True)

        self.add_to_training_outputs(outputs) # add to the list of outputs
        
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps