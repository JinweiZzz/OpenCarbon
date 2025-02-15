# 构图
import argparse
import math
import os
import pickle
import random
import time
from collections import OrderedDict

import cv2 as cv
import numpy as np
import setproctitle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn.functional import l1_loss, mse_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def poi_revise(m):
    return [m[0], m[1], m[2], m[3]+m[4]+m[5], m[6]+m[7]+m[8], m[9]+m[10], m[11], m[12], m[13], m[14], m[17], m[18], m[19], m[20], m[22]]


def read_data():
    tile2pic = {}
    tile2poi = {}
    tile2carbon = {}
    if args.city == 'yinchuan':
        train_tiles, valid_tiles, test_tiles = [], [], []
        tile2poi_img = np.load(
            'yinchuan_data/tile2poi_distribution.npy', allow_pickle=True).item()
        odiac = np.load('yinchuan_data/tile2carbon.npy',
                        allow_pickle=True).item()
        tile2poi_count = np.load(
            'yinchuan_data/tile2poi_count.npy', allow_pickle=True).item()
        data_dir = 'yinchuan_data/satellite_images_yinchuan/'
        filenames = os.listdir(data_dir)
        for filename in filenames:
            x_tile, y_tile = filename.split('_')[0], filename.split('_')[
                1].split('.')[0]
            if (int(x_tile), int(y_tile)) in tile2poi_count:
                im = cv.imread(data_dir+filename)
                tile2pic[(int(x_tile), int(y_tile))] = cv.resize(
                    im, (256, 256), interpolation=cv.INTER_AREA).transpose(2, 0, 1)
                tile2poi[(int(x_tile), int(y_tile))] = [
                    tile2poi_count[(int(x_tile), int(y_tile))]]
                if odiac[(int(x_tile), int(y_tile))] == 0:
                    continue
                tile2carbon[(int(x_tile), int(y_tile))
                            ] = np.log(odiac[(int(x_tile), int(y_tile))])
                train_tiles.append((int(x_tile), int(y_tile)))
    if args.city == 'beijing':
        district_list = ['dongcheng', 'xicheng', 'haidian',
                         'shijingshan', 'chaoyang', 'fengtai']
        train_tiles, valid_tiles, test_tiles = [], [], []
        tile2poi_img = np.load(
            'beijing_data/tile2poi_distribution_beijing.npy', allow_pickle=True).item()
        odiac = np.load(
            'beijing_data/til2carbon_beijing.npy', allow_pickle=True).item()
        tile2poi_count = np.load(
            'beijing_data/tile2poi_count_beijing.npy', allow_pickle=True).item()
        for district in district_list:
            data_dir = 'beijing_data/satellite_images_beijing/' + district + '/'
            filenames = os.listdir(data_dir)
            for filename in filenames:
                x_tile, y_tile = filename.split('_')[0], filename.split('_')[
                    1].split('.')[0]
                if (int(x_tile), int(y_tile)) in tile2poi_count:
                    im = cv.imread(data_dir+filename)
                    tile2pic[(int(x_tile), int(y_tile))
                             ] = im.transpose(2, 0, 1)
                    if odiac[(int(x_tile), int(y_tile))] == 0:
                        continue
                    tile2carbon[(int(x_tile), int(y_tile))
                                ] = np.log(odiac[(int(x_tile), int(y_tile))])
                    train_tiles.append((int(x_tile), int(y_tile)))
    if args.city == 'london':
        train_tiles, valid_tiles, test_tiles = [], [], []
        tile2poi_img = np.load(
            'london_data/tile2poi_distribution_london.npy', allow_pickle=True).item()
        odiac = np.load('london_data/til2carbon_london.npy',
                        allow_pickle=True).item()
        tile2poi_count = np.load(
            'london_data/tile2poi_count_london.npy', allow_pickle=True).item()
        data_dir = 'london_data/satellite_images_london/'
        filenames = os.listdir(data_dir)
        for filename in filenames:
            x_tile, y_tile = filename.split('_')[0], filename.split('_')[
                1].split('.')[0]
            if (int(x_tile), int(y_tile)) in tile2poi_count:
                im = cv.imread(data_dir+filename)
                tile2pic[(int(x_tile), int(y_tile))] = im.transpose(2, 0, 1)
                tile2poi[(int(x_tile), int(y_tile))] = [
                    tile2poi_count[(int(x_tile), int(y_tile))]]
                if odiac[(int(x_tile), int(y_tile))] == 0:
                    continue
                tile2carbon[(int(x_tile), int(y_tile))
                            ] = np.log(odiac[(int(x_tile), int(y_tile))])
                train_tiles.append((int(x_tile), int(y_tile)))
    if args.city == 'west_midland':
        train_tiles, valid_tiles, test_tiles = [], [], []
        tile2poi_img = np.load(
            'transfer_experiments/west_midland/tile2poi_distribution_west_midland.npz', allow_pickle=True)['poi_distribution'].item()
        odiac = np.load(
            'transfer_experiments/west_midland/til2carbon_west_midland.npy', allow_pickle=True).item()
        data_dir = 'transfer_experiments/west_midland/satellite_images_west_midland/'
        filenames = os.listdir(data_dir)
        for filename in filenames:
            x_tile, y_tile = filename.split('_')[0], filename.split('_')[
                1].split('.')[0]
            if (int(x_tile), int(y_tile)) in tile2poi_img:
                im = cv.imread(data_dir+filename)
                tile2pic[(int(x_tile), int(y_tile))] = im.transpose(2, 0, 1)
                if odiac[(int(x_tile), int(y_tile))] == 0:
                    continue
                tile2carbon[(int(x_tile), int(y_tile))] = np.log(
                    odiac[(int(x_tile), int(y_tile))])
                train_tiles.append((int(x_tile), int(y_tile)))
    if args.city == 'south_yorkshire':
        train_tiles, valid_tiles, test_tiles = [], [], []
        tile2poi_img = np.load(
            'transfer_experiments/south_yorkshire/tile2poi_distribution_south_yorkshire.npz', allow_pickle=True)['poi_distribution'].item()
        odiac = np.load(
            'transfer_experiments/south_yorkshire/til2carbon_south_yorkshire.npy', allow_pickle=True).item()
        data_dir = 'transfer_experiments/south_yorkshire/satellite_images_south_yorkshire/'
        filenames = os.listdir(data_dir)
        for filename in filenames:
            x_tile, y_tile = filename.split('_')[0], filename.split('_')[
                1].split('.')[0]
            if (int(x_tile), int(y_tile)) in tile2poi_img:
                im = cv.imread(data_dir+filename)
                tile2pic[(int(x_tile), int(y_tile))] = im.transpose(2, 0, 1)
                if odiac[(int(x_tile), int(y_tile))] == 0:
                    continue
                tile2carbon[(int(x_tile), int(y_tile))] = np.log(
                    odiac[(int(x_tile), int(y_tile))])
                train_tiles.append((int(x_tile), int(y_tile)))
    if args.city == 'manchester_cities':
        train_tiles, valid_tiles, test_tiles = [], [], []
        tile2poi_img = np.load(
            'transfer_experiments/manchester_cities/tile2poi_distribution_manchester_cities.npz', allow_pickle=True)['poi_distribution'].item()
        odiac = np.load(
            'transfer_experiments/manchester_cities/til2carbon_manchester_cities.npy', allow_pickle=True).item()
        data_dir = 'transfer_experiments/manchester_cities/satellite_images_manchester_cities/'
        filenames = os.listdir(data_dir)
        for filename in filenames:
            x_tile, y_tile = filename.split('_')[0], filename.split('_')[
                1].split('.')[0]
            if (int(x_tile), int(y_tile)) in tile2poi_img:
                try:
                    im = cv.imread(data_dir+filename)
                    tile2pic[(int(x_tile), int(y_tile))
                             ] = im.transpose(2, 0, 1)
                    if odiac[(int(x_tile), int(y_tile))] == 0:
                        continue
                    tile2carbon[(int(x_tile), int(y_tile))] = np.log(
                        odiac[(int(x_tile), int(y_tile))])
                    train_tiles.append((int(x_tile), int(y_tile)))
                except:
                    print(x_tile, y_tile)
    return tile2pic, tile2carbon, tile2poi_img, train_tiles, valid_tiles, test_tiles


def weights_init_1(m):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.nn.init.xavier_uniform_(m.weight, gain=1)


def weights_init_2(m):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.nn.init.xavier_uniform_(m.weight, gain=1)
    torch.nn.init.constant_(m.bias, 0)


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = F.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)


class HeadAttention(nn.Module):
    def __init__(self, dim_in: int, dim_k: int, dim_v: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_k)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_v)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:        # N*1*rep_dim
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_k: int, dim_v: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [HeadAttention(dim_in, dim_k, dim_v) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_v, dim_in)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(Attention, self).__init__()

        self.l1 = torch.nn.Linear(in_size, hidden_size, bias=True)
        self.ac = nn.Tanh()
        self.l2 = torch.nn.Linear(int(hidden_size), 1, bias=False)

        weights_init_2(self.l1)
        weights_init_1(self.l2)

    def forward(self, z):
        w = self.l1(z)
        w = self.ac(w)
        w = self.l2(w)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1)


class SE(nn.Module):
    def __init__(self, in_chnls, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls//ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls//ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return torch.sigmoid(out)


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, se_signal, ratio=4, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel),
        )

        self.se_signal = se_signal
        if self.se_signal:
            self.se = SE(outchannel, ratio)
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        if self.se_signal is True:
            coefficient = self.se(out)
            out *= coefficient
        residual = x if self.right is None else self.right(x)  # 检测右边直连的情况
        out += residual
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, input_dim, output_dim, se=False, layer_num=[3, 4, 6, 3]):
        super(ResNet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(input_dim, 64, 7, 1, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.se = se
        # 重复的 layer 分别有 3，4，6，3 个 residual block
        self.layer1 = self._make_layer(64, 64, layer_num[0], 4)
        self.layer2 = self._make_layer(64, 128, layer_num[1], 4)
        self.layer3 = self._make_layer(128, 256, layer_num[2], 4)
        self.layer4 = self._make_layer(256, 512, layer_num[3], 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, output_dim)

    def _make_layer(self, inchannel, outchannel, block_num, ratio, stride=2):
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel))

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel,
                      self.se, ratio, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel, 0, ratio))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16, act_layer=nn.ReLU, min_channels=8, reduction_channels=None,
                 gate_layer='sigmoid'):
        super(SEModule, self).__init__()
        reduction_channels = reduction_channels or max(
            channels // reduction, min_channels)
        self.fc1 = nn.Conv2d(channels, reduction_channels,
                             kernel_size=1, bias=True)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(reduction_channels, channels,
                             kernel_size=1, bias=True)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        x_se = x_se.sigmoid()
        return x * x_se

# beijing使用的encoder
# class POI_rep(nn.Module):
#     def __init__(self, channels, rep_dim, se=True):
#         super(POI_rep, self).__init__()
#         args.se = se
#         self.fc1 = nn.Conv2d(channels, 32, kernel_size=37,
#                              stride=3, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.ac1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(32, 1, kernel_size=37, stride=3,
#                              padding=1, bias=False)    # 28*28
#         self.fc = nn.Linear(196, rep_dim)
#         if args.se:
#             self.se1 = SEModule(32)

#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.bn1(out)
#         out = self.ac1(out)
#         if args.se:
#             out = self.se1(out)
#         out = self.fc2(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out

# London等英国城市用的POI_encoder


class POI_rep(nn.Module):
    def __init__(self, channels, rep_dim, se=True):
        super(POI_rep, self).__init__()
        args.se = se
        self.fc1 = nn.Conv2d(channels, 32, kernel_size=7,
                             stride=3, padding=3, bias=False)   # 86*86
        self.bn1 = nn.BatchNorm2d(32)
        self.ac1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(32, 32, kernel_size=7, stride=3,
                             padding=1, bias=False)    # 28*28
        self.bn2 = nn.BatchNorm2d(32)
        self.ac2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Conv2d(32, 1, kernel_size=3, stride=3,
                             padding=1, bias=False)    # 10*10
        self.fc = nn.Linear(100, rep_dim)
        if args.se:
            self.se1 = SEModule(32)
            self.se2 = SEModule(32)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.ac1(out)
        if args.se:
            out = self.se1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.ac2(out)
        if args.se:
            out = self.se2(out)
        out = self.fc3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Grid_rep(nn.Module):
    def __init__(self, dim_heads, dim_poi, dim_poi_input, se=True):
        super(Grid_rep, self).__init__()
        dim_image = dim_poi

        # self.image_encoder = ResNet(3, dim_image, False)
        self.image_encoder = models.resnet18(pretrained=False)
        checkpoint = torch.load('resnet18-5c106cde.pth')
        self.image_encoder.load_state_dict(checkpoint)
        self.image_encoder.fc = nn.Linear(
            self.image_encoder.fc.in_features, dim_poi)

        self.poi_encoder = POI_rep(dim_poi_input, dim_poi, se)
        # self.poi_encoder = ResNet(dim_poi_input, dim_poi, True, [2, 2, 2, 2])

    def forward(self, image_input, poi_input):
        image_rep = self.image_encoder(image_input)
        poi_rep = self.poi_encoder(poi_input)
        return image_rep, poi_rep


class Neighbor_rep(nn.Module):
    def __init__(self, dim_poi):
        super(Neighbor_rep, self).__init__()
        dim_image = dim_poi

        self.image_encoder = nn.Sequential(
            nn.Conv2d(dim_poi, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.image_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_fc = nn.Linear(128, dim_poi)

        self.poi_encoder = nn.Sequential(
            nn.Conv2d(dim_poi, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.poi_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.poi_fc = nn.Linear(128, dim_poi)

    def forward(self, image_input, poi_input):
        image_rep = self.image_encoder(image_input)
        image_rep = self.image_pool(image_rep)
        image_rep = torch.flatten(image_rep, 1)
        image_rep = self.image_fc(image_rep)
        poi_rep = self.poi_encoder(poi_input)
        poi_rep = self.poi_pool(poi_rep)
        poi_rep = torch.flatten(poi_rep, 1)
        poi_rep = self.poi_fc(poi_rep)
        return image_rep, poi_rep


class Regressor(nn.Module):
    def __init__(self, dim_heads, dim_poi, dim_poi_input, size=3, device='cpu', sat_matrix=None, poi_matrix=None, se=True):
        super(Regressor, self).__init__()
        dim_image = dim_poi
        self.dim_poi = dim_poi
        self.grid_rep = Grid_rep(dim_heads, dim_poi, dim_poi_input, se)
        self.neighbor_rep = Neighbor_rep(dim_poi)

        self.cross_from_neighborhood = MultiHeadAttention(
            dim_heads, 1, 1, 1)
        self.self_grid = MultiHeadAttention(
            dim_heads, 1, 1, 1)
        self.self_neighbor = MultiHeadAttention(dim_heads, 1, 1, 1)

        self.grid_agg = Attention(dim_poi)
        self.neighbor_agg = Attention(dim_poi)
        self.device = device

        self.image_input = torch.tensor(sat_matrix).float()
        self.poi_input = torch.tensor(poi_matrix).float()

        # self.regressor = nn.Sequential(nn.Linear(dim_poi, dim_poi),
        #                                nn.ReLU(),
        #                                nn.Linear(dim_poi, dim_poi),
        #                                nn.ReLU(),
        #                                nn.Linear(dim_poi, 1))

        # 加dropout防止过拟合
        self.regressor = nn.Sequential(
            nn.Linear(dim_poi*2, dim_poi),
            nn.ReLU(),
            nn.BatchNorm1d(dim_poi),
            nn.Linear(dim_poi, 1)
        )

    def self_attention(self, h, q, v):
        return self.self_grid(h, q, v) + v

    def neighbor_attention(self, h, q, v):
        return self.self_neighbor(h, q, v) + v

    def forward(self, neighbors):  # batch_size, 3, 256, 256
        batch_size = neighbors.shape[0]
        size = neighbors.shape[1]

        neighbors = neighbors.to(torch.int64)
        neighbors = neighbors.reshape(-1)
        image_input = self.image_input[neighbors].to(self.device)
        poi_input = self.poi_input[neighbors].to(self.device)

        grid_image_rep = torch.zeros(
            (batch_size*size*size, self.dim_poi)).to(self.device)
        grid_poi_rep = torch.zeros(
            (batch_size*size*size, self.dim_poi)).to(self.device)
        for mini_batch in range(size**2):
            grid_image_rep[mini_batch*batch_size:(mini_batch+1)*batch_size, :], grid_poi_rep[mini_batch*batch_size:(mini_batch+1)*batch_size, :] = self.grid_rep(
                image_input[mini_batch*batch_size:(mini_batch+1)*batch_size, :], poi_input[mini_batch*batch_size:(mini_batch+1)*batch_size, :])

        grid_image_rep = grid_image_rep.reshape(batch_size, size, size, -1)
        grid_poi_rep = grid_poi_rep.reshape(batch_size, size, size, -1)

        grid_image_emb = grid_image_rep[:, (size-1)//2, (size-1)//2, :]
        grid_poi_emb = grid_poi_rep[:, (size-1)//2, (size-1)//2, :]

        grid_rep = torch.stack([grid_image_emb, grid_poi_emb], dim=1)
        grid_rep = self.grid_agg(grid_rep)

        neighbor_image_rep = grid_image_rep.permute(0, 3, 1, 2)
        neighbor_poi_rep = grid_poi_rep.permute(0, 3, 1, 2)

        neighborhood_image_rep, neighborhood_poi_rep = self.neighbor_rep(
            neighbor_image_rep, neighbor_poi_rep)

        neighbor_rep = torch.stack(
            [neighborhood_image_rep, neighborhood_poi_rep], dim=1)
        neighbor_rep = self.neighbor_agg(neighbor_rep)

        neighbor_rep = neighbor_rep.unsqueeze(2)
        grid_rep = grid_rep.unsqueeze(2)
        grid_rep_self = self.self_attention(grid_rep, grid_rep, grid_rep)
        neighbor_rep_self = self.neighbor_attention(
            neighbor_rep, neighbor_rep, neighbor_rep)
        grid_rep_cross = self.cross_from_neighborhood(
            neighbor_rep_self, grid_rep_self, grid_rep_self)
        out = self.regressor(
            torch.cat([grid_rep_cross.squeeze(2), neighbor_rep_self.squeeze(2)], dim=1))
        return out, grid_image_emb, grid_poi_emb


class Regressor_grid(nn.Module):
    def __init__(self, dim_heads, dim_poi, dim_poi_input, size=3, device='cpu', sat_matrix=None, poi_matrix=None, se=True):
        super(Regressor_grid, self).__init__()
        dim_image = dim_poi
        self.dim_poi = dim_poi
        self.grid_rep = Grid_rep(dim_heads, dim_poi, dim_poi_input, se)

        self.self_grid = MultiHeadAttention(
            dim_heads, 1, 1, 1)

        self.grid_agg = Attention(dim_poi)
        self.device = device

        self.image_input = torch.tensor(sat_matrix).float()
        self.poi_input = torch.tensor(poi_matrix).float()

        # 加dropout防止过拟合
        self.regressor = nn.Sequential(
            nn.Linear(dim_poi, dim_poi),
            nn.ReLU(),
            nn.BatchNorm1d(dim_poi),
            nn.Linear(dim_poi, 1)
        )

    def self_attention(self, h, q, v):
        return self.self_grid(h, q, v) + v

    def forward(self, neighbors):  # batch_size, 3, 256, 256
        size = neighbors.shape[1]

        neighbors = neighbors.to(torch.int64)
        idxs = neighbors[:, (size-1)//2, (size-1)//2]
        image_input = self.image_input[idxs].to(self.device)
        poi_input = self.poi_input[idxs].to(self.device)

        grid_image_emb, grid_poi_emb = self.grid_rep(image_input, poi_input)
        grid_rep = torch.stack([grid_image_emb, grid_poi_emb], dim=1)
        grid_rep = self.grid_agg(grid_rep)

        # grid_rep = grid_rep.unsqueeze(2)
        out = self.regressor(grid_rep)
        return out, grid_image_emb, grid_poi_emb


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 tiles,
                 tile2idx,
                 neighbor_matrix,
                 tile2target):
        self.idxs = [tile2idx[tuple(tile)] for tile in tiles]
        self.targets = [tile2target[tuple(tile)] for tile in tiles]
        self.neighbor_matrix = neighbor_matrix[self.idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.targets[idx], self.neighbor_matrix[idx]


def contrastive_loss_cal(image_features, poi_features, temperature=0.07, device='cpu'):
    similarity_matrix = torch.matmul(image_features, poi_features.T)
    labels = torch.arange(image_features.size(0)).to(device)
    loss = nn.CrossEntropyLoss()(similarity_matrix / temperature, labels)
    return loss


parser = argparse.ArgumentParser(description='Carbon_Prediction')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1234)')
parser.add_argument('--device', default='cuda:0',
                    help='which device to use')
parser.add_argument('--dvs', default='gpu',
                    help='Wheter this is running on cpu or gpu')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--model_name_suffix', type=str,
                    default='10', help='exp_name')
parser.add_argument('--weight_decay', type=float, default=5e-3)
parser.add_argument('--city', type=str, default='newyork', help='exp_name')
parser.add_argument('--neighbor_size', type=int, default=1)
parser.add_argument('--contrastive', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--contrastive_epoch', type=int, default=100)
parser.add_argument('--ablation', type=str, default='none')
args = parser.parse_args()
if args.model_name_suffix == '10':
    args.model_name_suffix = ''.join(
        random.sample(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e'], 8))
writer = SummaryWriter('contrastive_carbon_{}/{}_{}_{}_{}_{}_{}_{}'.format(args.city, time.strftime("%m-%d %H:%M:%S", time.localtime()), str(args.contrastive), str(args.seed),
                       str(args.lr), str(args.batch_size), str(args.ablation), str(args.alpha)))
writer.add_scalar('lr', args.lr)
writer.add_scalar('batch_size', args.batch_size)
writer.add_scalar('neighbor_size', args.neighbor_size)
writer.add_text('model_name_suffix', args.model_name_suffix)
writer.add_text('ablation', args.ablation)
neighbor_size = args.neighbor_size

setproctitle.setproctitle('pred')

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(args.seed)

tile2pic, tile2carbon, tile2poi_img, train_tiles, valid_tiles, test_tiles = read_data()
# with open('train_test_split_london_south_east.pkl', 'rb') as f:
# if args.city == 'london':
#     with open('train_test_split_london_south_north.pkl', 'rb') as f:
#         train_tiles, valid_tiles, test_tiles = pickle.load(f)
# elif args.city == 'beijing':
#     with open('train_test_split_beijing_fengtai.pkl', 'rb') as f:
#         train_tiles, valid_tiles, test_tiles = pickle.load(f)
# elif args.city == 'yinchuan':
#     with open('train_test_split_yinchuan.pkl', 'rb') as f:
#         train_tiles, valid_tiles, test_tiles = pickle.load(f)
# else:
#     tiles = train_tiles + valid_tiles + test_tiles
#     train_tiles, test_tiles = train_test_split(
#         tiles, test_size=0.2, random_state=args.seed)
#     train_tiles, valid_tiles = train_test_split(
#         train_tiles, test_size=0.25, random_state=args.seed)
#     print(len(train_tiles), len(valid_tiles), len(test_tiles))

if args.city == 'beijing':
    with open('train_test_split_beijing_fengtai.pkl', 'rb') as f:
        train_tiles, valid_tiles, test_tiles = pickle.load(f)
elif args.city == 'yinchuan':
    with open('train_test_split_yinchuan.pkl', 'rb') as f:
        train_tiles, valid_tiles, test_tiles = pickle.load(f)
else:
    tiles = train_tiles + valid_tiles + test_tiles
    train_tiles, test_tiles = train_test_split(
        tiles, test_size=0.2, random_state=args.seed)
    train_tiles, valid_tiles = train_test_split(
        train_tiles, test_size=0.25, random_state=args.seed)
    print(len(train_tiles), len(valid_tiles), len(test_tiles))

tiles = list(tile2pic.keys())
tile2idx = dict(zip(tiles, list(range(len(tile2pic)))))
tiles = np.array(tiles)
min_x_tile, max_x_tile, min_y_tile, max_y_tile = min(
    tiles[:, 0]), max(tiles[:, 0]), min(tiles[:, 1]), max(tiles[:, 1])
poi_dim = np.shape(list(tile2poi_img.values())[0])[2]

count = 0
img_batch_matrix = np.zeros([len(tile2pic), 3, 256, 256])
poi_batch_matrix = np.zeros([len(tile2pic), poi_dim, 256, 256])
neighbor_matrix = np.zeros(
    [len(tile2pic), 2*neighbor_size+1, 2*neighbor_size+1])
for idx, tile in enumerate(tiles):
    x_tile, y_tile = tile[0], tile[1]
    img_batch_matrix[idx, :, :, :] = tile2pic[(x_tile, y_tile)]
    poi_batch_matrix[idx, :, :, :] = np.transpose(
        tile2poi_img[(x_tile, y_tile)], [2, 0, 1])
    for x in range(x_tile-neighbor_size, x_tile+neighbor_size+1):
        for y in range(y_tile-neighbor_size, y_tile+neighbor_size+1):
            if (x, y) in tile2idx:
                count += 1
                neighbor_matrix[idx, x-x_tile+neighbor_size,
                                y-y_tile+neighbor_size] = tile2idx[(x, y)]
print('img_batch_count:', count)
print('tiles num:', len(tiles))
assert count != 0


# count = 0
# img_batch_matrix = np.zeros([len(tile2pic),2*neighbor_size+1, 2*neighbor_size+1, 3, 256, 256])
# for idx, tile in enumerate(tiles):
#     x_tile, y_tile = tile[0], tile[1]
#     for x in range(x_tile-neighbor_size, x_tile+neighbor_size+1):
#         for y in range(y_tile-neighbor_size, y_tile+neighbor_size+1):
#             if (x, y) in tile2pic:
#                 count += 1
#                 img_batch_matrix[idx, x-x_tile+neighbor_size, y-y_tile+neighbor_size, :, :, :] = tile2pic[(x, y)]
# print('img_neighborhood_count:', count)
# assert count != 0

# count = 0
# poi_batch_matrix = np.zeros([len(tile2pic), 2*neighbor_size+1, 2*neighbor_size+1, poi_dim, 256, 256])
# for idx, tile in enumerate(tiles):
#     x_tile, y_tile = tile[0], tile[1]
#     for x in range(x_tile-neighbor_size, x_tile+neighbor_size+1):
#         for y in range(y_tile-neighbor_size, y_tile+neighbor_size+1):
#             if (x, y) in tile2pic:
#                 count += 1
#                 poi_batch_matrix[idx, x-x_tile+neighbor_size, y-y_tile+neighbor_size, :, :, :] = np.transpose(tile2poi_img[(x, y)], [2,0,1])
# print('poi_neighborhood_count:', count)
# assert count != 0


train_dataset = Dataset(train_tiles, tile2idx, neighbor_matrix, tile2carbon)
valid_dataset = Dataset(valid_tiles, tile2idx, neighbor_matrix, tile2carbon)
test_dataset = Dataset(test_tiles, tile2idx, neighbor_matrix, tile2carbon)
print(len(train_dataset), len(valid_dataset), len(test_dataset))

num_heads = 2
if args.ablation == 'se':
    model = Regressor(num_heads, 32, poi_dim, device=args.device,
                      sat_matrix=img_batch_matrix, poi_matrix=poi_batch_matrix, se=False)
elif args.ablation == 'neighbor':
    model = Regressor_grid(num_heads, 32, poi_dim, device=args.device,
                           sat_matrix=img_batch_matrix, poi_matrix=poi_batch_matrix)
else:
    model = Regressor(num_heads, 32, poi_dim, device=args.device,
                      sat_matrix=img_batch_matrix, poi_matrix=poi_batch_matrix, se=True)
model = model.to(args.device)
optimizer = optim.Adam(model.parameters(), lr=args.lr,
                       weight_decay=args.weight_decay)

train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=True,
                                           worker_init_fn=args.seed)
valid_loader = torch.utils.data.DataLoader(valid_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False,
                                           worker_init_fn=args.seed)
test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False,
                                          worker_init_fn=args.seed)


def loss_cal(output, target):
    return l1_loss(output, target, reduction='sum')


def mean_absolute_percentage_error(preds, labels):
    mask = labels != 0
    return np.fabs((labels[mask]-preds[mask])/labels[mask]).mean()


best_r2 = -10000
patience = 0
n_step, t_step = 0, 0
for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    running_main_loss, running_contrastive_loss = 0, 0
    outputs = []
    targets = []

    # for data_temp in tqdm(train_loader):
    for i, data_temp in enumerate(train_loader):
        target, neighbors = data_temp
        target, neighbors = target.float(), neighbors.float()
        if args.dvs == 'gpu':
            target = target.to(args.device)
        #     sat, poi, target, neighbors = sat.to(args.device), poi.to(args.device), target.to(args.device), neighbors.to(args.device)

        output, grid_image_rep, grid_poi_rep = model(neighbors)
        output = output.squeeze(1)
        optimizer.zero_grad()
        main_loss = loss_cal(output, target)
        contrastive_loss = contrastive_loss_cal(
            grid_image_rep, grid_poi_rep, device=args.device)
        if args.contrastive == 1 and epoch > args.contrastive_epoch:
            loss = main_loss + contrastive_loss*args.alpha
        else:
            loss = main_loss
        # print(loss_cal(output, target).item(), contrastive_loss(grid_image_rep, grid_poi_rep, device=args.device).item())
        targets += list(target.cpu().detach().numpy())
        outputs += list(output.cpu().detach().numpy())

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_main_loss += main_loss.item()
        running_contrastive_loss += contrastive_loss.item()

    mae = mean_absolute_error(outputs, targets)
    rmse = np.sqrt(mean_squared_error(outputs, targets))
    r2 = r2_score(targets, outputs)
    print('Train Epoch: {} [{}/ \tLoss: {:.6f}, MAE: {:.3f}, RMSE:{:.3f}, R2:{:.2f}'.format(
        epoch, len(train_tiles), running_loss/len(train_tiles), mae, rmse, r2))
    writer.add_scalar('train_loss', running_loss/len(train_tiles), n_step)
    writer.add_scalar('train_main_loss', running_main_loss /
                      len(train_tiles), n_step)
    writer.add_scalar('train_contrastive_loss',
                      running_contrastive_loss/len(train_tiles), n_step)
    writer.add_scalar('train_mae', mae, n_step)
    writer.add_scalar('train_rmse', rmse, n_step)
    writer.add_scalar('train_r2', r2, n_step)
    n_step += 1

    # validation
    if epoch % 5 == 0:
        model.eval()
        running_loss = 0.0
        running_main_loss, running_contrastive_loss = 0, 0
        outputs, targets = [], []

        for data_temp in tqdm(valid_loader):
            target, neighbors = data_temp
            target, neighbors = target.float(), neighbors.float()
            if args.dvs == 'gpu':
                target = target.to(args.device)
                output, grid_image_rep, grid_poi_rep = model(neighbors)
                output = output.squeeze(1)
                main_loss = loss_cal(output, target)
                contrastive_loss = contrastive_loss_cal(
                    grid_image_rep, grid_poi_rep, device=args.device)
                if args.contrastive == 1 and epoch > args.contrastive_epoch:
                    loss = main_loss + contrastive_loss*args.alpha
                else:
                    loss = main_loss
                targets += list(target.cpu().detach().numpy())
                outputs += list(output.cpu().detach().numpy())
                running_loss += loss.item()
                running_main_loss += main_loss.item()
                running_contrastive_loss += contrastive_loss.item()

        r_2 = r2_score(targets, outputs)
        mae = mean_absolute_error(outputs, targets)
        mape = mean_absolute_percentage_error(outputs, targets)
        rmse = np.sqrt(mean_squared_error(outputs, targets))
        print('Validation Epoch: {} [{}/ \tLoss: {:.6f}, MAE: {:.2f}, RMSE :{:.3f}, R2: {:.2f}, MAPE: {:.3f}'.format(
            epoch, len(valid_tiles), running_loss/len(valid_tiles), mae, rmse, r_2, mape))
        writer.add_scalar('val_loss', running_loss/len(valid_tiles), t_step)
        writer.add_scalar('val_main_loss', running_main_loss /
                          len(valid_tiles), t_step)
        writer.add_scalar('val_contrastive_loss',
                          running_contrastive_loss/len(valid_tiles), t_step)
        writer.add_scalar('val_mae',  mae, t_step)
        writer.add_scalar('val_rmse', rmse, t_step)
        writer.add_scalar('val_R2', r_2, t_step)
        writer.add_scalar('val_mape', mape, t_step)
        t_step += 1

        if r_2 > best_r2:
            best_r2 = r_2
            best_epoch = epoch
            fname = 'models/IJCAI_version/regressor_{}.pt'.format(
                args.model_name_suffix)
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, fname, _use_new_zipfile_serialization=False)
        else:
            patience += 1
            if patience > args.patience and epoch > 100:
                break

# test
targets, outputs = [], []
loss, running_loss = 0, 0
running_main_loss, running_contrastive_loss = 0, 0
model.eval()
checkpoint = torch.load(
    'models/IJCAI_version/regressor_{}.pt'.format(args.model_name_suffix))
model.load_state_dict(checkpoint['model_state_dict'])

for data_temp in test_loader:
    target, neighbors = data_temp
    target, neighbors = target.float(), neighbors.float()
    if args.dvs == 'gpu':
        target = target.to(args.device)
        output, grid_image_rep, grid_poi_rep = model(neighbors)
        output = output.squeeze(1)
        main_loss = loss_cal(output, target)
        contrastive_loss = contrastive_loss_cal(
            grid_image_rep, grid_poi_rep, device=args.device)
        if args.contrastive == 1 and epoch > args.contrastive_epoch:
            loss = main_loss + contrastive_loss*args.alpha
        else:
            loss = main_loss
        targets += list(target.cpu().detach().numpy())
        outputs += list(output.cpu().detach().numpy())
        running_loss += loss.item()
        running_main_loss += main_loss.item()
        running_contrastive_loss += contrastive_loss.item()

r_2 = r2_score(targets, outputs)
rmse = np.sqrt(mean_squared_error(outputs, targets))
mape = mean_absolute_percentage_error(outputs, targets)
mae = mean_absolute_error(outputs, targets)
print('Test Epoch: {} [{}/ \tLoss: {:.6f}, MAE: {:.2f}, RMSE :{:.3f}, R2: {:.2f}, MAPE: {:.3f}'.format(
    epoch, len(test_tiles), running_loss/len(test_tiles), mae, rmse, r_2, mape))
writer.add_scalar('test_loss', loss/len(test_tiles))
writer.add_scalar('test_main_loss', running_main_loss/len(test_tiles))
writer.add_scalar('test_contrastive_loss',
                  running_contrastive_loss/len(test_tiles))
writer.add_scalar('test_mae', mae)
writer.add_scalar('test_rmse', rmse)
writer.add_scalar('test_mape', mape)
writer.add_scalar('test_R2', r_2)
writer.close()
