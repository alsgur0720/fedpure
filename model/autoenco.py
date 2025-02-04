import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#from .visualizations import Ax3DPose
import sys
from einops import rearrange
import pickle
import torch
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # [batch, 16, 150, 13]
            nn.BatchNorm2d(16),
            # nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # [batch, 32, 75, 7]
            nn.BatchNorm2d(32),
            # nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [batch, 64, 38, 4]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64 * 38 * 4, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 128),
            nn.LeakyReLU()
        )
        

        self.decoder = nn.Sequential(
            nn.Linear(128, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 64 * 38 * 4),
            nn.LeakyReLU(),
            nn.Unflatten(1, (64, 38, 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [batch, 32, 75, 8]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # [batch, 16, 150, 16]
            nn.BatchNorm2d(16),
            # nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),  # [batch, 3, 300, 25]
        )
        
        
        self.data_bn = nn.BatchNorm1d(2 * 3 * 25) #num_person * in_channels * num_point


    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        
        restored_skeleton = decoded[:N * M, :C, :T, :V]
        restored_skeleton = restored_skeleton.view(N, M, C, T, V).permute(0, 2, 3, 4, 1)
        
        
        return restored_skeleton

class CustomDataset(Dataset):
    def __init__(self, data, labels):

        self.data = data
        self.labels = labels

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx], self.labels[idx]
    
    

class CustomDataset2(Dataset):
    def __init__(self, data, data2, labels):
        self.data = data
        self.data2 = data2
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data2[idx],self.labels[idx]
    

   
class DeepAutoencoder(nn.Module):
    def __init__(self):
        super(DeepAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            ResidualBlock(3, 16, stride=2),
            ResidualBlock(16, 32, stride=2),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            nn.Flatten(),
            nn.Linear(128 * 19 * 2, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 128 * 19 * 2),
            nn.LeakyReLU(),
            nn.Unflatten(1, (128, 19, 2)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        restored_skeleton = decoded[:N * M, :C, :T, :V]
        restored_skeleton = restored_skeleton.view(N, M, C, T, V).permute(0, 2, 3, 4, 1)

        return restored_skeleton

class DeepAutoencoder_connect(nn.Module):
    def __init__(self):
        super(DeepAutoencoder_connect, self).__init__()
 
        self.encoder1 = ResidualBlock(3, 16, stride=2)
        self.encoder2 = ResidualBlock(16, 32, stride=2)
        self.encoder3 = ResidualBlock(32, 64, stride=2)
        self.encoder4 = ResidualBlock(64, 128, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 19 * 2, 1024)
        #self.fc2 = nn.Linear(1024, 256)
       # self.fc3 = nn.Linear(256, 64)
        

       # self.fc4 = nn.Linear(64, 256)
        #self.fc5 = nn.Linear(256, 1024)
        self.fc6 = nn.Linear(1024, 128 * 19 * 2)
        self.unflatten = nn.Unflatten(1, (128, 19, 2))
        
        self.decoder4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder3 = nn.ConvTranspose2d(192, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder2 = nn.ConvTranspose2d(96, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder1 = nn.ConvTranspose2d(48, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final_layer = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        x = self.flatten(enc4)
        x = self.relu(self.fc1(x))
        #x = self.relu(self.fc2(x))
        #x = self.relu(self.fc3(x))
        

        #x = self.relu(self.fc4(x))
        #x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.unflatten(x)

        x = torch.cat([x, enc4], dim=1)
        x = self.relu(self.bn4(self.decoder4(x)))

        x = torch.cat([x, enc3], dim=1)
        x = self.relu(self.bn3(self.decoder3(x)))

        enc2 = F.pad(enc2, (0, x.size(3) - enc2.size(3), 0, x.size(2) - enc2.size(2)))

        x = torch.cat([x, enc2], dim=1)
        x = self.relu(self.bn2(self.decoder2(x)))
        enc1 = F.pad(enc1, (0, x.size(3) - enc1.size(3), 0, x.size(2) - enc1.size(2)))
        x = torch.cat([x, enc1], dim=1)
        x = self.relu(self.bn1(self.decoder1(x)))
        
        x = self.final_layer(x)
        
        restored_skeleton = x[:N * M, :C, :T, :V]
        restored_skeleton = restored_skeleton.view(N, M, C, T, V).permute(0, 2, 3, 4, 1)

        return restored_skeleton
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.residual(x) + self.shortcut(x)
        out = self.leaky_relu(out)
        return out
    
class Purifier(nn.Module):
    def __init__(self):
        super(Purifier, self).__init__()
        
        self.num_cls = 60

        self.encoder1 = ResidualBlock(3, 16, stride=2)
        self.encoder2 = ResidualBlock(16, 32, stride=2)
        self.encoder3 = ResidualBlock(32, 64, stride=2)
        self.encoder4 = ResidualBlock(64, 128, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 2, 1024)
        self.fc1rev = nn.Linear(512, 128 * 4 * 2)
        self.unflatten = nn.Unflatten(1, (128, 4, 2))
        
        self.decoder4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder3 = nn.ConvTranspose2d(192, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder2 = nn.ConvTranspose2d(96, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder1 = nn.ConvTranspose2d(48, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final_layer = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.relu = nn.LeakyReLU()
        self.cls = nn.Linear(512, self.num_cls)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        x = self.flatten(enc4)
        shared_x = self.relu(self.fc1(x))
        x_recon = shared_x[:, :512] 
        x_class = shared_x[:, 512:]
        # N*M,C,T,V
        c_new = x_class.size(1)
        x_class = x_class.view(N, M, c_new, -1)
        x_class = x_class.mean(3).mean(1)
        cls_output = self.cls(x_class)
        

        x = self.relu(self.fc1rev(x_recon))
        x = self.unflatten(x)

        x = torch.cat([x, enc4], dim=1)
        x = self.relu(self.bn4(self.decoder4(x)))

        x = torch.cat([x, enc3], dim=1)
        x = self.relu(self.bn3(self.decoder3(x)))

        enc2 = F.pad(enc2, (0, x.size(3) - enc2.size(3), 0, x.size(2) - enc2.size(2)))
        x = torch.cat([x, enc2], dim=1)
        x = self.relu(self.bn2(self.decoder2(x)))

        enc1 = F.pad(enc1, (0, x.size(3) - enc1.size(3), 0, x.size(2) - enc1.size(2)))
        x = torch.cat([x, enc1], dim=1)
        x = self.relu(self.bn1(self.decoder1(x)))
        
        x = self.final_layer(x)
        
        restored_skeleton = x[:N * M, :C, :T, :V]
        restored_skeleton = restored_skeleton.view(N, M, C, T, V).permute(0, 2, 3, 4, 1)
        
        return cls_output, restored_skeleton, shared_x


    
class Purifier_base(nn.Module):
    def __init__(self):
        super(Purifier_base, self).__init__()
        
        self.num_cls = 60

        self.encoder1 = ResidualBlock(3, 16, stride=2)
        self.encoder2 = ResidualBlock(16, 32, stride=2)
        self.encoder3 = ResidualBlock(32, 64, stride=2)
        self.encoder4 = ResidualBlock(64, 128, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 2, 1024)
        self.fc1rev = nn.Linear(1024, 128 * 4 * 2)
        self.unflatten = nn.Unflatten(1, (128, 4, 2))
        
        self.decoder4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder3 = nn.ConvTranspose2d(192, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder2 = nn.ConvTranspose2d(96, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder1 = nn.ConvTranspose2d(48, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final_layer = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        x = self.flatten(enc4)
        x = self.relu(self.fc1(x))
        

        x = self.relu(self.fc1rev(x))
        x = self.unflatten(x)

        x = torch.cat([x, enc4], dim=1)
        x = self.relu(self.bn4(self.decoder4(x)))

        x = torch.cat([x, enc3], dim=1)
        x = self.relu(self.bn3(self.decoder3(x)))

        enc2 = F.pad(enc2, (0, x.size(3) - enc2.size(3), 0, x.size(2) - enc2.size(2)))
        x = torch.cat([x, enc2], dim=1)
        x = self.relu(self.bn2(self.decoder2(x)))

        enc1 = F.pad(enc1, (0, x.size(3) - enc1.size(3), 0, x.size(2) - enc1.size(2)))
        x = torch.cat([x, enc1], dim=1)
        x = self.relu(self.bn1(self.decoder1(x)))
        
        x = self.final_layer(x)
        
        restored_skeleton = x[:N * M, :C, :T, :V]
        restored_skeleton = restored_skeleton.view(N, M, C, T, V).permute(0, 2, 3, 4, 1)
        
        return restored_skeleton
    
    
    
class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                  drop_out=0, adaptive=True):
        super(Model, self).__init__()
        
        self.num_cls = 60
 
        self.encoder1 = ResidualBlock(3, 16, stride=2)
        self.encoder2 = ResidualBlock(16, 32, stride=2)
        self.encoder3 = ResidualBlock(32, 64, stride=2)
        self.encoder4 = ResidualBlock(64, 128, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 2, 1024)
        self.fc1rev = nn.Linear(512, 128 * 4 * 2)
        self.unflatten = nn.Unflatten(1, (128, 4, 2))
        
        self.decoder4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder3 = nn.ConvTranspose2d(192, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder2 = nn.ConvTranspose2d(96, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder1 = nn.ConvTranspose2d(48, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final_layer = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.relu = nn.LeakyReLU()
        self.cls = nn.Linear(512, self.num_cls)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        x = self.flatten(enc4)
        shared_x = self.relu(self.fc1(x))
        x_recon = shared_x[:, :512] 
        x_class = shared_x[:, 512:]
        # N*M,C,T,V
        c_new = x_class.size(1)
        x_class = x_class.view(N, M, c_new, -1)
        x_class = x_class.mean(3).mean(1)
        cls_output = self.cls(x_class)
        

        x = self.relu(self.fc1rev(x_recon))
        x = self.unflatten(x)

        x = torch.cat([x, enc4], dim=1)
        x = self.relu(self.bn4(self.decoder4(x)))

        x = torch.cat([x, enc3], dim=1)
        x = self.relu(self.bn3(self.decoder3(x)))

        enc2 = F.pad(enc2, (0, x.size(3) - enc2.size(3), 0, x.size(2) - enc2.size(2)))
        x = torch.cat([x, enc2], dim=1)
        x = self.relu(self.bn2(self.decoder2(x)))

        enc1 = F.pad(enc1, (0, x.size(3) - enc1.size(3), 0, x.size(2) - enc1.size(2)))
        x = torch.cat([x, enc1], dim=1)
        x = self.relu(self.bn1(self.decoder1(x)))
        
        x = self.final_layer(x)
        
        restored_skeleton = x[:N * M, :C, :T, :V]
        restored_skeleton = restored_skeleton.view(N, M, C, T, V).permute(0, 2, 3, 4, 1)
        return restored_skeleton
        #return cls_output, restored_skeleton, shared_x
