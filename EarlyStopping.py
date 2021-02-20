'''
Author: your name
Date: 2021-02-05 08:36:32
LastEditTime: 2021-02-05 10:59:55
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /pytorch-AAPR/EarlyStopping.py
'''
import sys
import torch
from args import *


class EarlyStopping():
    def __init__(self, estimator, max_steps_without_increase, run_every_steps=1, higher_is_better=True, min_steps=0, max_steps=sys.maxsize):
        self.estimator = estimator
        self.max_steps_without_increase = max_steps_without_increase
        self.run_every_steps = run_every_steps
        self.higher_is_better = higher_is_better
        self.min_steps = min_steps
        self.max_steps = max_steps
        if higher_is_better:
            self.best_metrics = 0
        else:
            self.best_metrics = sys.maxsize
        self.best_index = 0
        self.run_steps = 0
        self.bad_steps = 0

    def need_to_stop(self, estimator, new_metric):
        self.run_steps += 1
        if self.run_steps < self.min_steps or (self.run_steps - self.min_steps) % self.run_every_steps != 0:
            return False
        if self.run_steps > self.max_steps:
            print('Stop for max_steps')
            return True

        if (self.higher_is_better and new_metric > self.best_metrics) or (not self.higher_is_better and new_metric < self.best_metrics):
            self.bad_steps = 0
            self._save_model(estimator)
            self.best_metrics = new_metric
            self.best_index = self.run_steps
        else:
            self.bad_steps += 1

        if self.bad_steps > self.max_steps_without_increase:
            print(f'Stop for early stop, at steps {self.best_index}')
            return True

        return False

    def _save_model(self, estimator):
        torch.save(self.estimator, opt.weight_datapath + "model.pt")
        torch.save(self.estimator.state_dict(), opt.weight_datapath + './state.pt')

# TODO 是否需要传模型？
