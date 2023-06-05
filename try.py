import os
import os.path
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method

from tqdm import tqdm
from copy import deepcopy
import argparse
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def print_args(logger, args):
    opt = vars(args)
    logger.cprint('------------ Options -------------')
    for k, v in sorted(opt.items()):
        logger.cprint('%s: %s' % (str(k), str(v)))
    logger.cprint('-------------- End ----------------\n')


def init_logger(log_dir, args):
    mkdir(log_dir)
    log_file = os.path.join(log_dir, 'log_%s.txt' % args.method)
    logger = IOStream(log_file)
    # logger.cprint(str(args))
    ## print arguments in format
    print_args(logger, args)
    return logger


def set_gpu(gpu):
    gpu_list = [int(x) for x in gpu.split(',')]
    print('use gpu:', gpu_list)
    return gpu_list.__len__()


def str2loss(args):

    if args.loss == "Cross_Entropy":
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        raise NotImplementedError("Loss \"" + args.loss + "\" not yet implemented")


def str2model(args):
    if args.model == "MLP":
        # from models.mlp import MLP
        return MLP(args)

    else:
        raise NotImplementedError("Model \"" + args.model + "\" not yet implemented")


def noise_injection(features, noise_factor):
    """噪声注入"""
    noise = torch.randn_like(features) * noise_factor
    return features + noise

def flip_data(data):
    p = torch.rand(1)
    if p < 0.5:
        return data
    flipped_data = torch.flip(data, dims=[0])
    return flipped_data

def shuffle_data(data):
    p = torch.rand(1)
    if p < 0.2:
        return data
    data = data.split(50, dim=0)
    data = [d for d in data]
    random.shuffle(data)
    # data.reverse()
    shuffled_data = torch.cat(data, dim=0)
    return shuffled_data 

class VideoDataset(Dataset):
    def __init__(self, data_path, label_path=None, train=True, transform=None, val_ratio=0.2):
        self.data_path = data_path
        self.transform = transform
        self.train = train

        self.file_list = sorted(os.listdir(self.data_path))
        n_samples = len(self.file_list)
        split_idx = int(n_samples * val_ratio)
        if train:
            self.file_list = self.file_list[split_idx:]
            with open(label_path, 'r') as f:
                labels = eval(f.read())
                self.labels = labels
        else:
            if label_path is not None:
                self.file_list = self.file_list[:split_idx]
                with open(label_path, 'r') as f:
                    labels = eval(f.read())
                self.labels = labels
            else:
                self.file_list = self.file_list
                self.labels = None


    def __getitem__(self, index):
        data = torch.from_numpy(np.load(os.path.join(self.data_path, self.file_list[index]))).squeeze()
        if self.labels is not None:
            label = int(self.labels[self.file_list[index]])
        else:
            label = -1  # use dummy label for test set

        if self.transform:
            # data = flip_data(data)
            # data = shuffle_data(data)
            data = noise_injection(data, noise_factor=0.1)

        return data, label

    def __len__(self):
        return len(self.file_list)

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.d_input = args.d_input
        self.n_input = args.n_input
        self.output_size = args.num_classes

        self.lstm = nn.LSTM(input_size=self.d_input, hidden_size=self.d_input//4, num_layers=1, batch_first=True)

        self.layers = nn.ModuleList()
        for i in range(5):
            layer = torch.nn.Sequential(
                torch.nn.Sequential(
                    nn.Linear(self.d_input//4, self.d_input//8),
                    nn.BatchNorm1d(50),
                    nn.GELU(),
                    nn.Dropout(p=0.15)
                ),
                torch.nn.Sequential(
                    nn.Linear(self.d_input//8, self.d_input//16),
                    nn.BatchNorm1d(50),
                    nn.GELU(),
                    nn.Dropout(p=0.1)
                ),
            )
            self.layers.append(layer)
        
        self.classifier1 = torch.nn.Sequential(
                nn.Linear(self.d_input//16, self.d_input//32),
                nn.BatchNorm1d(5),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(self.d_input//32, self.d_input//64)
            )

        self.classifier2 = nn.Linear(self.d_input//64, self.output_size)

        for module in self.layers:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
        for module in self.classifier1:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)

    def forward(self, x, val=False):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        h0 = torch.zeros(1, batch_size, self.d_input//4).to(x.device)
        c0 = torch.zeros(1, batch_size, self.d_input//4).to(x.device)
        x, _ = self.lstm(x, (h0, c0))

        data = x.split(50, dim=1)
        data = [d for d in data]
        out = []
        for i in range(5):
            x_i = self.layers[i](data[i])
            x_i = x_i.mean(dim=1).unsqueeze(1)
            out.append(x_i)
        x = torch.cat(out, dim=1)
        x = self.classifier1(x)
        x = x.mean(dim=1)
        logit = self.classifier2(x).squeeze(-1)
        return logit

def train(args):
    args.device = device
    logger = init_logger(args.log_dir, args)

    train_dataset = VideoDataset(args.data_path, args.label_path, train=True, transform=True)
    val_dataset = VideoDataset(args.data_path, args.label_path, train=False)
    test_dataset = VideoDataset(args.to_be_predicted, train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=12, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,num_workers=12, shuffle=False, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=12, shuffle=False, pin_memory=True)

    model = str2model(args)
    # summary(model, (250, 2048))
    model = nn.DataParallel(model, device_ids=list(range(args.num_gpu)))
    model = model.to(device)
    loss_fn = str2loss(args)
    loss_fn.to(device)

    start_epoch = 1
    if args.resume > 0:
        logger.cprint(f'Resume previous training, start from epoch {args.resume}, loading previous model')
        start_epoch = args.resume
        resume_model_path = os.path.join(args.checkpoint_path, f'best_model.pth')

        if os.path.exists(resume_model_path):
            model.load_state_dict(torch.load(resume_model_path)['model'])
            for name, param in model.named_parameters():
                if "layer" in name:
                    param.requires_grad = False
                if 'classifer' in name:
                    param.requires_grad = False
            print(model.module.classifer[0].weight)

        else:
            raise RuntimeError(f'{resume_model_path} does not exist, loading failed')


    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0, last_epoch=-1)


    # Start Traning process--------------------------------------
    model.train()
    train_batch_id = 0
    best_epoch = 0
    best_acc = 0.
    sep = 1e-6


    for epoch in range(start_epoch, args.epochs + 1):

        logger.cprint(f'----------Start Training Epoch-[{epoch}/{args.epochs}]------------')

        ttl_rec = 0.
        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            # perturbed_inputs = fast_gradient_method(model, inputs, 0.1, np.inf)
            optimizer.zero_grad()
            logits = model(inputs)
            # if epoch >= 20:
            #     adv_logits = model(perturbed_inputs)
            #     logits = (logits + adv_logits) / 2.
            loss = loss_fn(logits, targets)

            loss.backward()
            optimizer.step()

            batch_rec = loss
            ttl_rec += batch_rec

            train_batch_id += 1

            if (i + 1) % 100 == 0:
                logger.cprint(f'Training Batch-[{i + 1}/{len(train_dataloader)}]:{batch_rec:.5f}')

        epoch_rec = ttl_rec / len(train_dataloader)
        logging = f'Training results for epoch -- {epoch}: Epoch_Rec:{epoch_rec}'
        logger.cprint(logging)
        scheduler.step()
        model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_dataloader, desc='Epoch:{:d} val'.format(epoch)):
                inputs = inputs.to(device)
                targets = targets.to(device)

                logits = model(inputs, val=True)
                predicted = torch.argmax(logits, dim=-1)
                total_correct += (predicted == targets).sum().item()
                total_samples += inputs.size(0)

        # 计算准确率
        accuracy = total_correct / total_samples
        logger.cprint(f'accuracy = {accuracy}')


        if accuracy > best_acc:
            best_acc = accuracy
            best_epoch = epoch

            logger.cprint('******************Model Saved******************')
            save_dict = {
                'model': deepcopy(model.state_dict()),
                'best_epoch': best_epoch,
                'best_acc': best_acc
            }
            torch.save(save_dict, os.path.join(args.log_dir, 'best_model.pth'))

        logger.cprint(f'best_epoch = {best_epoch}, best_acc = {best_acc}')

        model.train()

    model.eval()
    model.load_state_dict(torch.load(os.path.join(args.log_dir, 'best_model.pth'))['model'])
    predicted_result = []
    with torch.no_grad():
        for inputs , _ in tqdm(test_dataloader, desc='Epoch:{:d} test'.format(epoch)):
            inputs = inputs.to(device)
            logits = model(inputs, val=False)
            predicted = torch.argmax(logits, dim=-1)

            predicted_list = predicted.tolist()
            predicted_result.extend(predicted_list)

    with open('video_dataset/submit_example.txt', 'r') as f:
        example = eval(f.read())

    temp_idx = 0
    for key, _ in example.items():
        example[key] = predicted_result[temp_idx]
        temp_idx+=1

    with open(f'{args.log_dir}/example.txt', 'w', encoding='utf-8') as f:
        json.dump(example, f)

    model.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='first_stage')
    parser.add_argument('--data_path', type=str, required=False, default='video_dataset/train/train_feature',
                            help='Path to the numpy data to be train;')
    parser.add_argument('--label_path', type=str, required=False, default='video_dataset/train/train_list.txt',
                            help='Path to the label file ;')
    parser.add_argument('--to_be_predicted', type=str, required=False, default='video_dataset/test_A/test_A_feature',
                            help='Path to the numpy data to_be_predicted ;')


    parser.add_argument('--method', type=str, default='baseline')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size 1024 for yelp, 256 for amazon.')

    parser.add_argument('--lr', type=float, default=1e-4,
                            help='Initial learning rate. [0.1 for amazon and 0.001 for yelp]')
    parser.add_argument('--workers', type=int, default=-1,
                            help='number of workers')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD Momentum')
    parser.add_argument('--resume', type=int, default=-1,help='Flag to resume training [default: -1];')
    parser.add_argument('--is_train', type=int, default=-1,help='Flag to resume training [default: -1];')


    # lr_scheduler StepLR
    parser.add_argument('--step_size', type=float, default=20, help='Decay learning rate every step_size epoches [default: 50];')
    parser.add_argument('--gamma', type=float, default=0.5, help='lr decay')
    # optimizer
    parser.add_argument('--SGD', action='store_true', help='Flag to use SGD optimizer')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')

    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
    parser.add_argument('--seed', type=int, default=3047, help='Random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')

    # model pth
    parser.add_argument('--save_path', type=str, default='video_results', help='Directory to the save log and checkpoints')
    parser.add_argument('--extra_info', type=str, default='', help='Extra information in save_path')

    # model
    parser.add_argument('--model', type=str, default='MLP',help='Name of model to use')
    # loss
    parser.add_argument('--loss', type=str, default='Cross_Entropy',help='Name of loss to use')

    parser.add_argument('--num_classes', type=int, default=5, help='The number of class')
    parser.add_argument('--n_input', type=int, default=250, help='The number of the input feature')
    parser.add_argument('--d_input', type=int, default=2048, help='The dimension of the input feature')


    # args = parser.parse_args()
    args = parser.parse_known_args()[0]
    set_seed(args.seed)
    args.num_gpu = set_gpu(args.gpu)
    args.log_dir = args.save_path + f"/{args.dataset}/{args.method}/{args.model}/bs{args.batch_size}_ep{args.epochs}_lr{args.lr:.4f}_step{args.step_size}_gm{args.gamma:.1f}"

    timestamp = time.strftime('%m_%d_%H_%M')
    if args.extra_info:
        args.log_dir = args.log_dir + '/' + args.extra_info + '_' + timestamp
    else:
        args.log_dir = args.log_dir + '/' + timestamp

    train(args)
