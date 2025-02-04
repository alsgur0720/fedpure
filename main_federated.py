#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob
import wandb
# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm
from einops import rearrange
# from torchlight import DictAction

from torch.utils.data import Subset

import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from model.autoenco import Purifier
# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

indicies_10 = []

torso = np.array([1, 2])-1
indicies_10.append(torso)
head = np.array([3, 4, 21])-1
indicies_10.append(head)

right_arm = np.array([9, 10])-1
indicies_10.append(right_arm)
right_hand = np.array([11, 12, 24, 25])-1
indicies_10.append(right_hand)

left_arm = np.array([5, 6])-1
indicies_10.append(left_arm)
left_hand = np.array([7, 8, 22, 23])-1
indicies_10.append(left_hand)


right_leg = np.array([17, 18])-1
indicies_10.append(right_leg)
right_foot = np.array([19, 20])-1
indicies_10.append(right_foot)


left_leg = np.array([13, 14])-1
indicies_10.append(left_leg)
left_foot = np.array([15, 16])-1
indicies_10.append(left_foot)


indicies_5 = []

torso = np.array([1, 2, 3, 4, 21])-1
indicies_5.append(torso)

right_arm = np.array([5, 6, 7, 8, 22, 23])-1
indicies_5.append(right_arm)

left_arm = np.array([9, 10, 11, 12, 23, 24])-1
indicies_5.append(left_arm)

right_leg = np.array([17, 18, 19, 20])-1
indicies_5.append(right_leg)

left_leg = np.array([13, 14, 15, 16])-1
indicies_5.append(left_leg)

def calculate_centroid(points):
    return np.mean(points, axis=0)

def translation_matrix(tx, ty, tz):
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])

def scaling_matrix(sx, sy, sz):
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])


def rotation_matrix_from_points(src_points, dst_points):
    H = np.dot(src_points.T, dst_points)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    return R

def affine_transform_matrix(src_points, dst_points):
    src_centroid = calculate_centroid(src_points)
    dst_centroid = calculate_centroid(dst_points)

    T1 = translation_matrix(-src_centroid[0], -src_centroid[1], -src_centroid[2])
    T2 = translation_matrix(dst_centroid[0], dst_centroid[1], dst_centroid[2])

    src_points_centered = src_points - src_centroid
    dst_points_centered = dst_points - dst_centroid

    src_norm = np.linalg.norm(src_points_centered, axis=0)
    dst_norm = np.linalg.norm(dst_points_centered, axis=0)

    S = scaling_matrix(dst_norm[0] / src_norm[0], dst_norm[1] / src_norm[1], dst_norm[2] / src_norm[2])

    
    
    R2 = rotation_matrix_from_points(src_points_centered, dst_points_centered)

    
    R4x4 = np.eye(4)
    R4x4[:3, :3] = R2

    A = T2 @ R4x4 @ T1
    return A

class DictAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(DictAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        input_dict = eval(f'dict({values})')  #pylint: disable=W0123
        output_dict = getattr(namespace, self.dest)
        for k in input_dict:
            output_dict[k] = input_dict[k]
        setattr(namespace, self.dest, output_dict)


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/test_flask',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd120-cross-set/default.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='test', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--save-epoch',
        type=int,
        default=30,
        help='the start epoch to save model (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=0,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)

    parser.add_argument('--num_clients', type=int, required=True, help='Total number of clients')
    parser.add_argument('--client_id', type=int, required=True, help='Client ID')
    
    return parser



class FederatedClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device[0] if type(device) is list else device
        self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
        self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
        self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=0.1,
                momentum=0.9,
                nesterov=True,
                weight_decay=0.0004)
        
        self.global_step = 0
        
        self.loss = nn.CrossEntropyLoss().cuda(self.device)
        self.lr = arg.base_lr
        wandb.init(project="Federated-SAR_adam")
        wandb.watch(self.model, log="all")
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        
        self.model.load_state_dict(state_dict, strict=True)

        # Save aggregated weights to file
        torch.save(state_dict, 'federated_weight/aggregated_weights_momentum.pt')

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        # if True:
        #     with open('{}/log.txt'.format(self.work_dir), 'a') as f:
        #         print(str, file=f)
    
    
    def adjust_learning_rate(self, epoch):
        if True:
            if epoch < 5:
                lr = 0.1* (epoch + 1) / 5
            else:
                lr = 0.1* (
                        0.1 ** np.sum(epoch >= np.array([20, 40, 60])))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()
    
    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time
    
    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time
    
    def fit(self, parameters, reference_frame, lambda_mal, config):
        num_epoch = 1
        # self.set_parameters(parameters)
        self.model.train()
        self.print_log('Training epoch: {}'.format(num_epoch + 1))
        loader = self.train_loader
        self.adjust_learning_rate(num_epoch)

        loss_value = []
        acc_value = []
        self.train_writer.add_scalar('epoch', num_epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)
        
        
        for epoch in range(0, num_epoch):
            for batch_idx, (data, label, index) in enumerate(process):
                self.global_step += 1
                with torch.no_grad():
                    data = data.float().cuda(self.device)
                    label = label.long().cuda(self.device)
                timer['dataloader'] += self.split_time()

                if lambda_mal :
                    data = Purifier(data)
                
                reference_frame = rearrange(reference_frame[:,:,:,0], 'x y z -> y z x')
                data_t = rearrange(data[:,:,:,0], 'x y z -> y z x')
                A = affine_transform_matrix(reference_frame, data_t)
                reference_frame = reference_frame @ A[:3, :3].T + A[:3, 3]

                for k in range(0, len(indicies_10)):
                    reference_frame = reference_frame[0,indicies_10[k],:]
                    data_t = data_t[0,indicies_10[k],:]
                    transform_prototype = affine_transform_matrix(reference_frame, data_t)
                    
                # forward
                output = self.model(data)
                loss = self.loss(output, label)
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_value.append(loss.data.item())
                timer['model'] += self.split_time()

                value, predict_label = torch.max(output.data, 1)
                acc = torch.mean((predict_label == label.data).float())
                acc_value.append(acc.data.item())
                self.train_writer.add_scalar('acc', acc, self.global_step)
                self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)

                
                # statistics
                self.lr = self.optimizer.param_groups[0]['lr']
                self.train_writer.add_scalar('lr', self.lr, self.global_step)
                timer['statistics'] += self.split_time()

                wandb.log({"Train Loss": loss.data.item(), "Train Accuracy": acc.data.item(), "Learning Rate": self.lr}, step=self.global_step)
            # statistics of time consumption and loss
            proportion = {
                k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
                for k, v in timer.items()
            }
            self.print_log(
                '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_value), np.mean(acc_value)*100))
            self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

            if False:
                state_dict = self.model.state_dict()
                weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

                torch.save(weights, 'federated' + '-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')
        return self.get_parameters(config), len(self.train_loader.dataset), {}, transform_prototype

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0
        correct = 0
        process = tqdm(self.test_loader, ncols=40)
        with torch.no_grad():
            for batch_idx, (data, target, index) in enumerate(process):
                data = data.float().cuda(self.device)
                target = target.long().cuda(self.device)
                output = self.model(data)
                loss += nn.CrossEntropyLoss()(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
                wandb.log({"Test Loss": loss, "Test Accuracy": correct / len(self.test_loader.dataset)}, step=batch_idx)
        accuracy = correct / len(self.test_loader.dataset)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}
    

class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        # pdb.set_trace()
        self.load_model()
        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            self.load_data()
            
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        self.model = self.model.cuda(self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)

    def save_arg(self):
    # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)


    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
    
    def split_dataset(self, dataset, num_clients, client_id):
    # Determine the size of each subset
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        np.random.shuffle(indices)

        split_sizes = [dataset_size // num_clients] * num_clients
        for i in range(dataset_size % num_clients):
            split_sizes[i] += 1

        start_idx = sum(split_sizes[:client_id])
        end_idx = start_idx + split_sizes[client_id]

        subset_indices = indices[start_idx:end_idx]
        return Subset(dataset, subset_indices)
    
    def load_data(self):
        # print(type(self.arg.train_feeder_args), "train feeder args")
        # print(self.arg.test_feeder_args, "train feeder args")
        # exit()
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=self.split_dataset(Feeder(**self.arg.train_feeder_args), num_clients=self.arg.num_clients, client_id=self.arg.client_id),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=self.split_dataset(Feeder(**self.arg.test_feeder_args), num_clients=self.arg.num_clients, client_id=self.arg.client_id), 
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)
        # self.data_loader['test'] = torch.utils.data.DataLoader(
        #     dataset=Feeder(**self.arg.test_feeder_args), 
        #     batch_size=self.arg.test_batch_size,
        #     shuffle=False,
        #     num_workers=self.arg.num_worker,
        #     drop_last=False,
        #     worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args)
        print(self.model)
        # i = 1
        # for ndarray in self.model.parameters():
            # i = i + 1
        # print(i)
        # exit()
        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        
        if self.arg.weights:
            self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])
            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights, strict=False)
                for name, param in self.model.named_parameters():
                    if name == 'fc.weight' or name == 'fc.bias':
                        print(name)
                        param.requires_grad = True
                    else:
                        param.requires_grad = True
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

                


    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()
    
    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            client = FederatedClient(self.model, self.data_loader['train'], self.data_loader['test'], self.arg.device)
            fl.client.start_numpy_client(server_address="localhost:8100", client=client)
            

if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.full_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
