import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

import os
import time
import argparse

from dataset import PlayDataset
from model import Model

import warnings
warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def train(opt):
    net = Model(opt)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    train_dataset = PlayDataset(is_train=True, train_val=0.9, transform=transform_test)
    trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
    test_dataset = PlayDataset(is_train=False, train_val=0.9, transform=transform_test)
    testloader = DataLoader(test_dataset, batch_size=100, shuffle=True, num_workers=8)

    criterion = torch.nn.CTCLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    for epoch in range(100):
        print('\nEpoch: %d' % epoch)
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        net.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, sample_batch in enumerate(trainloader):
            inputs = sample_batch['image'].type(torch.FloatTensor).to(device)
            targets = sample_batch['label'].to(device)
            preds = net(inputs, None).log_softmax(2)
            preds_size = torch.IntTensor([preds.size(1)] * targets.shape[0])
            
            preds = preds.permute(1, 0, 2)
            length = torch.ones(targets.shape[0]) * 4
            cost = criterion(preds, targets, preds_size.to(device), length.to(device))
            net.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)  # gradient clipping with 5 (Default)
            optimizer.step()
            
            train_loss += cost.item()
            total += targets.size(0)
        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, sample_batch in enumerate(testloader):
                inputs = sample_batch['image'].type(torch.FloatTensor).to(device)
                targets = sample_batch['label'].to(device)
                preds = net(inputs, None).log_softmax(2)
                preds_size = torch.IntTensor([preds.size(1)] * targets.shape[0])
                
                preds = preds.permute(1, 0, 2)
                length = torch.ones(targets.shape[0]) * 4
                cost = criterion(preds, targets, preds_size.to(device), length.to(device))
                test_loss += cost.item()
                total += targets.size(0)

            sample = preds.permute(1, 0, 2)[0]
            _, predicted = sample.max(1)
            print(predicted)
            print(targets[0])
            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # save model
    state = {
            'net': net.state_dict(),
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/first.t7')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default="None", help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default="VGG", help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default="None", help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default="CTC", help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=3, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()
    opt.num_class = 63
    train(opt)
