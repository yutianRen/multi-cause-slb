import argparse
import math
import os
import random
import pickle
import time
import yaml
from collections import defaultdict, OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import matplotlib.pyplot as plt

from nets.mlp import MLP
from datasets.dataset import BasicDataset
from datasets.data_utils import get_data_loader

torch.set_float32_matmul_precision('high')

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7.0 / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    '''
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    '''

    def _lr_lambda(current_step):
        '''
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        '''

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


class Trainer():
    def __init__(self, arg, oid, iid, bid, type):

        self.arg = arg

        self.load_model()
        print('load model finished')
        self.load_param_groups()
        self.load_optimizer()
        self.load_lr_scheduler()

        self.global_step = 0
        self.lr = self.arg.base_lr
        self.best_acc = {'test': 0, 'val': 0}
        self.best_acc_epoch = {'test': 0, 'val': 0}
        self.acc_list = {'test': {}, 'val': {}}

        self.mean_loss_list = []
        self.mean_loss_val = []
        self.oid = oid
        self.iid = iid
        self.bid = bid
        self.type = type
        
       
        print('work dir exist: ', os.path.exists(arg.work_dir))

        if not os.path.exists(arg.work_dir):
            print('creating ', arg.work_dir)
            os.makedirs(arg.work_dir)
    def load_model(self):
        self.output_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = MLP(**self.arg.model_args).to(self.output_device)
        torch.compile(self.model)
        #self.model = nn.DataParallel(self.model).to(self.output_device)
        self.loss = nn.CrossEntropyLoss().cuda(self.output_device)
        # self.loss = nn.NLLLoss().to(self.output_device)

    def load_param_groups(self):
        """
        Template function for setting different learning behaviour
        (e.g. LR, weight decay) of different groups of parameters
        """
        self.param_groups = defaultdict(list)

        for name, params in self.model.named_parameters():
            self.param_groups['other'].append(params)

        self.optim_param_groups = {
            'other': {'params': self.param_groups['other']}
        }

    def load_optimizer(self):
        params = list(self.optim_param_groups.values())
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                params,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                params,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(
                params,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )

        else:
            raise ValueError('Unsupported optimizer: {}'.format(self.arg.optimizer))

    def load_lr_scheduler(self):
        # self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[250, 500], gamma=0.1)
        self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                            self.arg.num_epoch,
                                                            num_warmup_steps=self.arg.num_epoch * 0)

    def load_data(self, data_p, label_p, test_dp, test_lp, val_dp, val_lp):
        from nets.feeder import Feeder
        self.data_loader = dict()

        def worker_seed_fn(worker_id):
            # give workers different seeds
            return init_seed(self.arg.seed + worker_id + 1)


        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(data_p, label_p),
                batch_size=self.arg.batch_size,
                shuffle=True,
                drop_last=False,
                worker_init_fn=worker_seed_fn)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(test_dp, test_lp),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            drop_last=False,
            worker_init_fn=worker_seed_fn)

        self.data_loader['val'] = torch.utils.data.DataLoader(
            dataset=Feeder(val_dp, val_lp),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            drop_last=False,
            worker_init_fn=worker_seed_fn)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log(f'Local current time: {localtime}')

    def print_log(self, s, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = f'[ {localtime} ] {s}'
        print(s)
        if self.arg.print_log:
            with open(os.path.join(self.arg.work_dir, 'log.txt'), 'a') as f:
                print(s, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def save_states(self, epoch, states, out_folder, out_name):
        out_folder_path = os.path.join(self.arg.work_dir, out_folder)
        out_path = os.path.join(out_folder_path, out_name)
        os.makedirs(out_folder_path, exist_ok=True)
        torch.save(states, out_path)

    def save_weights(self, epoch, out_folder='weights'):
        state_dict = self.model.state_dict()
        weights = OrderedDict([
            [k.split('module.')[-1], v.cpu()]
            for k, v in state_dict.items()
        ])

        weights_name = f'weights-{epoch}-{int(self.global_step)}.pt'
        self.save_states(epoch, weights, out_folder, weights_name)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))

    def train(self, epoch, save_model=True):
        self.model.train()
        loader = self.data_loader['train']
        loss_values = []

        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        current_lr = self.optimizer.param_groups[0]['lr']
        # self.print_log(f'Training epoch: {epoch + 1}, LR: {current_lr:.4f}')

        # process = tqdm(loader, dynamic_ncols=True)
        for batch_idx, (data, label, index) in enumerate(loader):
            self.global_step += 1
            # get data
            with torch.no_grad():
                data = data.float().to(self.output_device)
                label = label.long().to(self.output_device)
            timer['dataloader'] += self.split_time()

            # backward
            self.optimizer.zero_grad()

            output = self.model(data)

            if isinstance(output, tuple):
                output, l1 = output
                l1 = l1.mean()
            else:
                l1 = 0

            loss = self.loss(output, label)

            loss.backward()

            loss_values.append(loss.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output, 1)
            acc = torch.mean((predict_label == label).float())

            #####################################

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
            self.optimizer.step()

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            timer['statistics'] += self.split_time()

            # Delete output/loss after each batch since it may introduce extra mem during scoping
            # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770/3
            del output
            del loss

        # statistics of time consumption and loss
        proportion = {
            k: f'{int(round(v * 100 / sum(timer.values()))):02d}%'
            for k, v in timer.items()
        }

        mean_loss = np.mean(loss_values)
        self.mean_loss_list.append(mean_loss)

        # PyTorch > 1.2.0: update LR scheduler here with `.step()`
        # and make sure to save the `lr_scheduler.state_dict()` as part of checkpoint
        self.lr_scheduler.step()

        if save_model:
            # save training checkpoint & weights
            self.save_weights(epoch + 1)
            # self.save_checkpoint(epoch + 1)

    def eval(self, epoch, save_score=False, loader_name=['test']):
        # Skip evaluation if too early

        if epoch + 1 < self.arg.eval_start:
            return

        with torch.no_grad():
            self.model = self.model.to(self.output_device)
            self.model.eval()
            # self.print_log(f'Eval epoch: {epoch + 1}')
            for ln in loader_name:
                loss_values = []
                score_batches = []
                step = 0
                # process = tqdm(self.data_loader[ln], dynamic_ncols=True)
                for batch_idx, (data, label, index) in enumerate(self.data_loader[ln]):
                    data = data.float().to(self.output_device)
                    label = label.long().to(self.output_device)

                    # torch.save(label, 'last_layer_lb_trad.pt')

                    output = self.model(data)
                    if isinstance(output, tuple):
                        output, l1 = output
                        l1 = l1.mean()
                    else:
                        l1 = 0
                    loss = self.loss(output, label)
                    score_batches.append(output.data.cpu().numpy())
                    loss_values.append(loss.item())

                    step += 1

                score = np.concatenate(score_batches)
                loss = np.mean(loss_values)
                if ln == 'val':
                    self.mean_loss_val.append(loss)
                accuracy = self.data_loader[ln].dataset.top_k(score, 1)
                self.acc_list[ln][epoch + 1] = accuracy
                if accuracy > self.best_acc[ln]:
                    self.best_acc[ln] = accuracy
                    self.best_acc_epoch[ln] = epoch + 1

        # Empty cache after evaluation
        torch.cuda.empty_cache()

    def start(self):
        if self.arg.phase == 'train':
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.train(epoch, save_model=False)
                self.eval(epoch, save_score=False, loader_name=['val', 'test'])

            with open(os.path.join(self.arg.work_dir, f'{self.oid}_{self.iid}_{self.bid}_{self.type}.txt'), 'w') as f:
                f.write(str(self.acc_list['test'][self.best_acc_epoch['val']]))

            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f"Current fold: {self.oid}, {self.iid}, {self.bid}")
            self.print_log(f"Best val accuracy: {self.best_acc['val']}")
            self.print_log(f"Epoch number: {self.best_acc_epoch['val']}")
            self.print_log(f"Test accuracy at this epoch: {self.acc_list['test'][self.best_acc_epoch['val']]}")

            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')

            self.best_acc = self.acc_list['test'][self.best_acc_epoch['val']]

        elif self.arg.phase == 'test':

            self.print_log(f'Model:   {self.arg.model}')

            self.eval(
                epoch=0,
                loader_name=['test']
            )

            self.print_log('Done.\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # load arg form config file
    p = parser.parse_args()
    with open('net_config.yaml', 'r') as f:
        default_arg = yaml.full_load(f)
    parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)

    torch_trainer = torchNN(arg)

    data = torch.ones(10, 1, 3)
    label = torch.ones(10)
    torch_trainer.load_data(data, label, data, label, data, label)

    torch_trainer.start()
