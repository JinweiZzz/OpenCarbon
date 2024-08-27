jj# 构图
import os
import numpy as np
import os
import torch
import cv2 as cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.functional import mse_loss, l1_loss
import argparse
import random
from torch.utils.tensorboard import SummaryWriter
import math
import setproctitle
from collections import OrderedDict
from torch import Tensor
import torch.nn.functional as f

def poi_revise(m):
    return [m[0], m[1], m[2], m[3]+m[4]+m[5], m[6]+m[7]+m[8], m[9]+m[10], m[11], m[12], m[13], m[14], m[17], m[18], m[19], m[20], m[22]]

def read_data():
    tile2pic = {}
    tile2poi = {}
    tile2carbon = {}
    if args.city == 'west_midland':
        train_tiles, valid_tiles, test_tiles = [], [], []
        tile2poi_img = np.load('west_midland/tile2poi_distribution_west_midland.npy', allow_pickle=True).item()
        odiac = np.load('west_midland/til2carbon_west_midland.npy', allow_pickle=True).item() 
        tile2poi_count = np.load('west_midland/tile2poi_count_west_midland.npy', allow_pickle=True).item()
        tile2vec = np.load('west_midland/tile2vec_resnet18_west_midland.npy', allow_pickle=True).item()
        data_dir = 'west_midland/satellite_images_west_midland/'
        filenames = os.listdir(data_dir)
        for filename in filenames:
            x_tile, y_tile = filename.split('_')[0], filename.split('_')[1].split('.')[0]
            if (int(x_tile), int(y_tile)) in tile2poi_count:
                im = cv.imread(data_dir+filename)
                tile2pic[(int(x_tile), int(y_tile))] = im.transpose(2, 0, 1)
                tile2poi[(int(x_tile), int(y_tile))] = [tile2poi_count[(int(x_tile), int(y_tile))]]
                tile2carbon[(int(x_tile), int(y_tile))] = np.log(max(odiac[(int(x_tile), int(y_tile))], 1))
                train_tiles.append((int(x_tile), int(y_tile)))
    if args.city == 'manchester_cities':
        train_tiles, valid_tiles, test_tiles = [], [], []
        tile2poi_img = np.load('manchester_cities/tile2poi_distribution_manchester_cities.npy', allow_pickle=True).item()
        odiac = np.load('manchester_cities/til2carbon_manchester_cities.npy', allow_pickle=True).item() 
        tile2poi_count = np.load('manchester_cities/tile2poi_count_manchester_cities.npy', allow_pickle=True).item()
        tile2vec = np.load('manchester_cities/tile2vec_resnet18_manchester_cities.npy', allow_pickle=True).item()
        data_dir = 'manchester_cities/satellite_images_manchester_cities/'
        filenames = os.listdir(data_dir)
        for filename in filenames:
            x_tile, y_tile = filename.split('_')[0], filename.split('_')[1].split('.')[0]
            if (int(x_tile), int(y_tile)) in tile2poi_count:
                try:
                    im = cv.imread(data_dir+filename)
                    tile2pic[(int(x_tile), int(y_tile))] = im.transpose(2, 0, 1)
                except:
                    print(filename)
                    continue
                tile2pic[(int(x_tile), int(y_tile))] = im.transpose(2, 0, 1)
                tile2poi[(int(x_tile), int(y_tile))] = [tile2poi_count[(int(x_tile), int(y_tile))]]
                tile2carbon[(int(x_tile), int(y_tile))] = np.log(max(odiac[(int(x_tile), int(y_tile))], 1))
                train_tiles.append((int(x_tile), int(y_tile)))
    if args.city == 'newyork':
        district_list = ['bronx', 'brooklyn', 'manhatton', 'queens', 'statan_island']
        year = '19'
        month = '12'
        train_tiles, valid_tiles, test_tiles = [], [], []
        tile2poi_img = np.load('newyork_data/tile2poi_distribution.npy', allow_pickle=True).item()
        odiac = np.load('newyork_data/odiac_new_york_1912.npy', allow_pickle=True).item() 
        tile2vec = np.load('newyork_data/main_model_v2/tile2vec_resnet18.npy', allow_pickle=True).item()
        for district in district_list:
            poi_dir = 'newyork_data/safegraph_poi/' + district +'/'
            data_dir = 'newyork_data/satellite_images/' + district +'/'
            tile2poi_checkin, tile2poi_count = np.load(poi_dir+'tile2poi_{}20{}-{}_combined.npy'.format(year, year, month), allow_pickle=True) 
            filenames = os.listdir(data_dir)
            for filename in filenames:
                x_tile, y_tile = filename.split('_')[1], filename.split('_')[2].split('.')[0]
                if (int(x_tile), int(y_tile)) in tile2poi_checkin:
                    im = cv.resize(cv.imread(data_dir+filename), (256, 256), interpolation=cv.INTER_AREA).transpose(2, 0, 1)
                    tile2pic[(int(x_tile), int(y_tile))] = im
                    tile2poi[(int(x_tile), int(y_tile))] = [poi_revise(tile2poi_count[(int(x_tile), int(y_tile))])]
                    if district == 'bronx':
                        valid_tiles.append((int(x_tile), int(y_tile)))
                    elif district == 'manhatton':
                        test_tiles.append((int(x_tile), int(y_tile)))
                    else:
                        train_tiles.append((int(x_tile), int(y_tile)))
                    tile2carbon[(int(x_tile), int(y_tile))] = np.log(max(odiac[(int(x_tile), int(y_tile))], 1))
    if args.city == 'beijing':
        district_list = ['dongcheng', 'xicheng', 'haidian', 'shijingshan', 'chaoyang', 'fengtai']
        train_tiles, valid_tiles, test_tiles = [], [], []
        tile2poi_img = np.load('beijing_data/tile2poi_distribution_bj.npy', allow_pickle=True).item()
        odiac = np.load('beijing_data/til2carbon_bj.npy', allow_pickle=True).item() 
        tile2poi_count = np.load('beijing_data/tile2poi_count_bj.npy', allow_pickle=True).item()
        tile2vec = np.load('beijing_data/tile2vec_resnet18.npy', allow_pickle=True).item()
        for district in district_list:
            data_dir = 'satellite_images_bj/' + district +'/'
            filenames = os.listdir(data_dir)
            for filename in filenames:
                x_tile, y_tile = filename.split('_')[0], filename.split('_')[1].split('.')[0]
                if (int(x_tile), int(y_tile)) in tile2poi_count:
                    im = cv.imread(data_dir+filename)
                    tile2pic[(int(x_tile), int(y_tile))] = im.transpose(2, 0, 1)
                    tile2poi[(int(x_tile), int(y_tile))] = [tile2poi_count[(int(x_tile), int(y_tile))]]
                    tile2carbon[(int(x_tile), int(y_tile))] = np.log(max(odiac[(int(x_tile), int(y_tile))], 1))
                    train_tiles.append((int(x_tile), int(y_tile)))
    if args.city == 'london':
        train_tiles, valid_tiles, test_tiles = [], [], []
        tile2poi_img = np.load('london_data/tile2poi_distribution_london.npy', allow_pickle=True).item()
        odiac = np.load('london_data/til2carbon_london.npy', allow_pickle=True).item() 
        tile2poi_count = np.load('london_data/tile2poi_count_london.npy', allow_pickle=True).item()
        tile2vec = np.load('london_data/tile2vec_resnet18_london.npy', allow_pickle=True).item()
        data_dir = 'london_data/satellite_images_london/'
        filenames = os.listdir(data_dir)
        for filename in filenames:
            x_tile, y_tile = filename.split('_')[0], filename.split('_')[1].split('.')[0]
            if (int(x_tile), int(y_tile)) in tile2poi_count:
                im = cv.imread(data_dir+filename)
                tile2pic[(int(x_tile), int(y_tile))] = im.transpose(2, 0, 1)
                tile2poi[(int(x_tile), int(y_tile))] = [tile2poi_count[(int(x_tile), int(y_tile))]]
                tile2carbon[(int(x_tile), int(y_tile))] = np.log(max(odiac[(int(x_tile), int(y_tile))], 1))
                train_tiles.append((int(x_tile), int(y_tile)))
    return tile2pic, tile2poi, tile2carbon, tile2poi_img, tile2vec, train_tiles, valid_tiles, test_tiles


def weights_init_1(m):
    seed=args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.nn.init.xavier_uniform_(m.weight,gain=1)
    
def weights_init_2(m):
    seed=args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.nn.init.xavier_uniform_(m.weight,gain=1)
    torch.nn.init.constant_(m.bias,0)


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(Attention, self).__init__()
        self.l1=torch.nn.Linear(in_size, hidden_size, bias=True)
        self.ac=nn.Tanh()
        self.l2=torch.nn.Linear(int(hidden_size), 1, bias=False)
        
        weights_init_2(self.l1)
        weights_init_1(self.l2)
        

    def forward(self, z):
        w=self.l1(z)
        
        w=self.ac(w)
        w=self.l2(w)
        
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
    def __init__(self, inchannel, outchannel, se_signal, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 7, stride, 3, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 7, 1, 3, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.se_signal = se_signal
        if self.se_signal:
            self.se = SE(outchannel, 4)
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        if self.se_signal is True:
            coefficient = self.se(out)
            out *= coefficient
        residual = x if self.right is None else self.right(x)  
        out += residual
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, poi_dim, output_dim, se=False):
        super(ResNet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(poi_dim, 64, 7, 1, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.se = se
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=1)
        self.layer3 = self._make_layer(128, 256, 6, stride=1)
        self.layer4 = self._make_layer(256, output_dim, 3, stride=1)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel))

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, self.se, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel, self.se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16, act_layer=nn.ReLU, min_channels=8, reduction_channels=None,
                 gate_layer='sigmoid'):
        super(SEModule, self).__init__()
        reduction_channels = reduction_channels or max(channels // reduction, min_channels)
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, bias=True)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        x_se = x_se.sigmoid()
        return x * x_se

class POI_rep(nn.Module):
    def __init__(self, channels, rep_dim):
        super(POI_rep, self).__init__()
        self.fc1 = nn.Conv2d(channels, 32, kernel_size=7, stride=3, padding=3, bias=False)   # 86*86
        self.se1 = SEModule(32)
        self.bn1 = nn.BatchNorm2d(32)
        self.ac1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(32, 32, kernel_size=7, stride=3, padding=1, bias=False)    # 28*28
        self.se2 = SEModule(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.ac2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Conv2d(32, 1, kernel_size=3, stride=3, padding=1, bias=False)    # 10*10
        self.fc = nn.Linear(100, rep_dim) 
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.ac1(out)
        out = self.se1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.ac2(out)
        out = self.se2(out)
        out = self.fc3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = f.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)

class HeadAttention(nn.Module):
    def __init__(self, dim_in: int, dim_k: int, dim_v: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_k)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_v)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
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

class Grid_Rep(torch.nn.Module):
    def __init__(self, poi_dim, img_rep_dim, num_heads):
        super(Grid_Rep, self).__init__()
        img_rep = models.resnet18(pretrained=False)
        checkpoint = torch.load('resnet18-5c106cde.pth')
        img_rep.load_state_dict(checkpoint)
        img_rep.fc = nn.Linear(img_rep.fc.in_features, img_rep_dim)
        self.img_rep = img_rep

        poi_rep = POI_rep(poi_dim, img_rep_dim)    
        self.poi_rep = poi_rep

        dim_model = img_rep_dim
        dim_k = dim_v = dim_model // num_heads

        self.img_attention = MultiHeadAttention(num_heads, dim_model, dim_k, dim_v)
        self.poi_attention = MultiHeadAttention(num_heads, dim_model, dim_k, dim_v)

        self.attention = Attention(in_size = img_rep_dim)

    def forward(self, sat, poi):
        sat_rep = self.img_rep(sat).unsqueeze(dim=1)
        poi_rep = self.poi_rep(poi).unsqueeze(dim=1)
        # cross attention
        sat_rep = self.img_attention(poi_rep, sat_rep, sat_rep)
        poi_rep = self.poi_attention(sat_rep, poi_rep, poi_rep)
        rep = torch.stack([sat_rep, poi_rep], dim=1)
        out = self.attention(rep).squeeze()
        return out  
    
class District_Rep(torch.nn.Module):
    def __init__(self, poi_dim, img_dim, rep_dim, num_heads):
        super(District_Rep, self).__init__()
        img_rep = ResNet(img_dim, rep_dim, False)
        poi_rep = ResNet(poi_dim, rep_dim, True)
        self.img_rep = img_rep
        self.poi_rep = poi_rep

        dim_model = rep_dim
        dim_k = dim_v = dim_model // num_heads

        self.img_attention = MultiHeadAttention(num_heads, dim_model, dim_k, dim_v)
        self.poi_attention = MultiHeadAttention(num_heads, dim_model, dim_k, dim_v)

        self.attention = Attention(in_size = rep_dim)

    def forward(self, sat_matrix, poi_matrix, tiles):
        sat_rep = self.img_rep(sat_matrix.unsqueeze(dim=0).float())
        poi_rep = self.poi_rep(poi_matrix.unsqueeze(dim=0).float())
        sat_rep = sat_rep[:, :, tiles[:, 0], tiles[:, 1]].permute(0, 2, 1)
        poi_rep = poi_rep[:, :, tiles[:, 0], tiles[:, 1]].permute(0, 2, 1)
        # cross attention
        sat_rep = self.img_attention(poi_rep, sat_rep, sat_rep)
        poi_rep = self.poi_attention(sat_rep, poi_rep, poi_rep)
        # attention
        rep = torch.stack([sat_rep, poi_rep], dim=1)
        out = self.attention(rep).squeeze()
        return out  

class Regressor(torch.nn.Module):
    def __init__(self, in_dim, num_heads=3):
        super(Regressor, self).__init__()
        dim_model = in_dim
        dim_k = dim_v = dim_model // num_heads
        self.cross_attention = MultiHeadAttention(num_heads, dim_model, dim_k, dim_v)
        self.model = nn.Sequential(nn.Linear(in_dim, 128),
                            nn.ReLU(),
                            nn.BatchNorm1d(128),
                            nn.Linear(128, 32),
                            nn.ReLU(),
                            nn.BatchNorm1d(32),
                            nn.Linear(32, 1))

    def forward(self, grid_rep, district_rep):
        grid_rep = self.cross_attention(district_rep, grid_rep, grid_rep) + grid_rep
        out = self.model(grid_rep)
        return out
        

parser = argparse.ArgumentParser(description='Carbon_Prediction')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1234)')
parser.add_argument('--device', default='cuda:0',
                    help='which device to use')
parser.add_argument('--dvs', default='gpu',
                    help='Wheter this is running on cpu or gpu')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--model_name_suffix', type=str, default='10', help='exp_name')
parser.add_argument('--weight_decay', type=float, default=5e-3)
parser.add_argument('--city', type=str, default='newyork', help='exp_name')
parser.add_argument('--M', type=int, default='neighbor_size', default=3)
args = parser.parse_args()
if args.model_name_suffix == '10':
        args.model_name_suffix = ''.join(
            random.sample(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e'], 8))
writer = SummaryWriter('main_model/{}_{}_{}'.format(args.city, str(args.lr), str(args.batch_size)))
writer.add_scalar('lr', args.lr)
writer.add_scalar('batch_size', args.batch_size)
writer.add_text('model_name_suffix', args.model_name_suffix)

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(args.seed)

tile2pic, tile2poi, tile2carbon, tile2poi_img, tile2vec, train_tiles, valid_tiles, test_tiles = read_data()
all_tile = train_tiles+valid_tiles+test_tiles
train_idxs, test_idxs = train_test_split(list(range(len(all_tile))), train_size=0.8, random_state=args.seed)
train_idxs, valid_idxs = train_test_split(train_idxs, train_size=0.7, random_state=args.seed)
train_tiles = np.array(all_tile)[train_idxs].tolist()
valid_tiles = np.array(all_tile)[valid_idxs].tolist()
test_tiles = np.array(all_tile)[test_idxs].tolist()

tiles = np.array(list(tile2pic.keys()))
min_x_tile, max_x_tile, min_y_tile, max_y_tile = min(tiles[:, 0]), max(tiles[:, 0]), min(tiles[:, 1]), max(tiles[:, 1])
poi_dim = len(list(tile2poi.values())[0][0])
img_pca_dim = 128

poi_matrix_district = np.zeros([max_x_tile-min_x_tile+1, max_y_tile-min_y_tile+1, poi_dim])
for tile in tile2poi:
    poi_matrix_district[tile[0]-min_x_tile, tile[1]-min_y_tile, :] = tile2poi[tile][0]
tmp = (np.max(poi_matrix_district, axis=(0, 1))-np.min(poi_matrix_district, axis=(0, 1)))
tmp = np.array([a if a != 0 else 1 for a in tmp.tolist()])
poi_matrix_district = (poi_matrix_district - np.min(poi_matrix_district, axis=(0, 1)))/tmp
poi_matrix_district = torch.from_numpy(poi_matrix_district)
poi_matrix_district = poi_matrix_district.permute(2, 0, 1) 

img_matrix_district = np.zeros([max_x_tile-min_x_tile+1, max_y_tile-min_y_tile+1, img_pca_dim])
for tile in tile2poi:
    img_matrix_district[tile[0]-min_x_tile, tile[1]-min_y_tile, :] = tile2vec[tile]
img_matrix_district = torch.from_numpy(img_matrix_district)
img_matrix_district = img_matrix_district.permute(2, 0, 1) 

img_matrix = np.zeros([len(tile2pic), 3, 256, 256])
idx = 0
for tile in tile2pic:
    img_matrix[idx, :, :, :] = tile2pic[tile]/256 # 归一化
    idx += 1
img_matrix = torch.from_numpy(img_matrix)

poi_distribution_matrix = np.zeros([len(tile2pic), poi_dim, 256, 256])
idx = 0
for tile in tile2pic:
    poi_distribution_matrix[idx, :, :, :] = tile2poi_img[tile].transpose(2, 0, 1)
    idx += 1
poi_distribution_matrix = torch.from_numpy(poi_distribution_matrix)

tile2idx = dict(zip(list(tile2pic.keys()), list(range(len(tile2pic)))))

carbon_target_matrix = np.zeros((max_x_tile-min_x_tile+1, max_y_tile-min_y_tile+1))
for tile in tile2carbon:
    carbon_target_matrix[(int(tile[0])-min_x_tile, int(tile[1])-min_y_tile)] = tile2carbon[tile]

train_tiles = [[tile[0]-min_x_tile, tile[1]-min_y_tile] for tile in train_tiles]
valid_tiles = [[tile[0]-min_x_tile, tile[1]-min_y_tile] for tile in valid_tiles]
test_tiles = [[tile[0]-min_x_tile, tile[1]-min_y_tile] for tile in test_tiles]

img_rep_dim = 256
num_heads = 4
grid_model = Grid_Rep(poi_dim, img_rep_dim=img_rep_dim, num_heads=num_heads)
district_model = District_Rep(poi_dim, img_pca_dim, img_rep_dim, num_heads)
regressor = Regressor(img_rep_dim)

grid_model, district_model, regressor = grid_model.to(args.device), district_model.to(args.device), regressor.to(args.device)
optimizer = optim.RMSprop(list(grid_model.parameters())+list(district_model.parameters())+list(regressor.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

train_loader = torch.utils.data.DataLoader(np.array(train_tiles), num_workers=0, batch_size=args.batch_size, shuffle=False,
                                        worker_init_fn=args.seed)
valid_loader = torch.utils.data.DataLoader(np.array(valid_tiles), num_workers=0, batch_size=args.batch_size, shuffle=False,
                                        worker_init_fn=args.seed)
test_loader = torch.utils.data.DataLoader(np.array(test_tiles), num_workers=0, batch_size=args.batch_size, shuffle=False,
                                        worker_init_fn=args.seed)

def loss_cal(output, target):
    return mse_loss(output, target, reduction='sum')

def mean_absolute_percentage_error(preds,labels): 
    mask=labels!=0
    return np.fabs((labels[mask]-preds[mask])/labels[mask]).mean() 


min_loss = 1e15
patience = 0
n_step, t_step = 0, 0
for epoch in range(args.epochs):
    grid_model.train()
    regressor.train()
    running_loss = 0.0
    outputs = []
    targets = []

    for data_temp in enumerate(train_loader):
        tiles = data_temp[1].numpy()
        tmp = tiles.tolist()
        tmp =  [(m[0]+min_x_tile, m[1]+min_y_tile) for m in tmp]
        idxs = [tile2idx[tuple(tile)] for tile in tmp]
        target = torch.tensor(carbon_target_matrix[tiles[:, 0], tiles[:, 1]].tolist()).unsqueeze(1)
        sat = img_matrix[idxs].float()
        poi = poi_distribution_matrix[idxs].float()
        if args.dvs == 'gpu':
            sat, poi, target = sat.to(args.device), poi.to(args.device), target.to(args.device)
            img_matrix_district, poi_matrix_district = img_matrix_district.to(args.device), poi_matrix_district.to(args.device)
        grid_rep = grid_model(sat, poi)
        district_rep = district_model(img_matrix_district, poi_matrix_district, tiles)
        output = regressor(grid_rep, district_rep)    
        optimizer.zero_grad()
        loss = loss_cal(output, target.float())
        targets += list(target.squeeze().cpu().detach().numpy())
        outputs += list(output.squeeze().cpu().detach().numpy())

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    mae = mean_absolute_error(outputs, targets)
    rmse = np.sqrt(mean_squared_error(outputs, targets))
    r2 = r2_score(targets, outputs)
    print('Train Epoch: {} [{}/ \tLoss: {:.6f}, MAE: {:.3f}, RMSE:{:.3f}, R2:{:.2f}'.format(epoch, len(train_tiles), running_loss/len(train_tiles), mae, rmse, r2))
    writer.add_scalar('train_loss', running_loss/len(train_tiles), n_step)
    writer.add_scalar('train_mae', mae, n_step)
    writer.add_scalar('train_rmse', rmse, n_step)
    writer.add_scalar('train_r2', r2, n_step)
    n_step += 1  

    if epoch % 5 == 0:
        grid_model.eval()
        regressor.eval()
        running_loss = 0.0
        outputs, targets = [], []

        for data_temp in enumerate(valid_loader):
            tiles = data_temp[1].numpy()
            tmp = tiles.tolist()
            tmp =  [(m[0]+min_x_tile, m[1]+min_y_tile) for m in tmp]
            idxs = [tile2idx[tuple(tile)] for tile in tmp]
            target = torch.tensor(carbon_target_matrix[tiles[:, 0], tiles[:, 1]].tolist()).unsqueeze(1)
            sat = img_matrix[idxs].float()
            poi = poi_distribution_matrix[idxs].float()
            if args.dvs == 'gpu':
                sat, poi, target = sat.to(args.device), poi.to(args.device), target.to(args.device)
                img_matrix_district, poi_matrix_district = img_matrix_district.to(args.device), poi_matrix_district.to(args.device)
            grid_rep = grid_model(sat, poi)
            district_rep = district_model(img_matrix_district, poi_matrix_district, tiles)
            output = regressor(grid_rep, district_rep)   
            loss = loss_cal(output, target.float())
            targets += list(target.squeeze().cpu().detach().numpy())
            outputs += list(output.squeeze().cpu().detach().numpy())
            running_loss += loss.item()

        r_2 = r2_score(targets, outputs)
        mae = mean_absolute_error(outputs, targets)
        mape = mean_absolute_percentage_error(outputs, targets)
        rmse = np.sqrt(mean_squared_error(outputs, targets))
        print('Validation Epoch: {} [{}/ \tLoss: {:.6f}, MAE: {:.2f}, RMSE :{:.3f}, R2: {:.2f}, MAPE: {:.3f}'.format(epoch, len(valid_tiles), running_loss/len(valid_tiles), mae, rmse, r_2, mape))
        writer.add_scalar('val_loss', running_loss/len(valid_tiles), t_step)
        writer.add_scalar('val_mae',  mae, t_step)
        writer.add_scalar('val_rmse', rmse, t_step)
        writer.add_scalar('val_R2', r_2, t_step)
        writer.add_scalar('val_mape', mape, t_step) 
        t_step += 1  

        if running_loss/len(valid_tiles) < min_loss:
                patience = 0
                min_loss = running_loss/len(valid_tiles)
                best_epoch = epoch
                fname = 'models/grid_model_{}.pt'.format(args.model_name_suffix)
                rname = 'models/regressor_{}.pt'.format(args.model_name_suffix)
                dname = 'models/district_model_{}.pt'.format(args.model_name_suffix)                
                torch.save({'model_state_dict': grid_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, fname, _use_new_zipfile_serialization=False)
                torch.save({'model_state_dict': district_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, dname, _use_new_zipfile_serialization=False)
                torch.save({'model_state_dict': regressor.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, rname, _use_new_zipfile_serialization=False)
        else:
            patience += 1
            if patience > args.patience:
                break

# test
targets = []
outputs = []
loss = 0.0
running_loss = 0
grid_model.eval()
regressor.eval()
checkpoint = torch.load('models/grid_model_{}.pt'.format(args.model_name_suffix))
grid_model.load_state_dict(checkpoint['model_state_dict'])
checkpoint = torch.load('models/regressor_{}.pt'.format(args.model_name_suffix))
regressor.load_state_dict(checkpoint['model_state_dict'])
checkpoint = torch.load('models/district_model_{}.pt'.format(args.model_name_suffix))
district_model.load_state_dict(checkpoint['model_state_dict'])
for data_temp in enumerate(test_loader):
    tiles = data_temp[1].numpy()
    tmp = tiles.tolist()
    tmp =  [(m[0]+min_x_tile, m[1]+min_y_tile) for m in tmp]
    idxs = [tile2idx[tuple(tile)] for tile in tmp]
    target = torch.tensor(carbon_target_matrix[tiles[:, 0], tiles[:, 1]].tolist()).unsqueeze(1)
    sat = img_matrix[idxs].float()
    poi = poi_distribution_matrix[idxs].float()
    if args.dvs == 'gpu':
        sat, poi, target = sat.to(args.device), poi.to(args.device), target.to(args.device)
        img_matrix_district, poi_matrix_district = img_matrix_district.to(args.device), poi_matrix_district.to(args.device)
    grid_rep = grid_model(sat, poi)
    district_rep = district_model(img_matrix_district, poi_matrix_district, tiles)
    output = regressor(grid_rep, district_rep)  
    loss = loss_cal(output, target.float())
    targets += list(target.squeeze().cpu().detach().numpy())
    outputs += list(output.squeeze().cpu().detach().numpy())
    running_loss += loss.item()

r_2 = r2_score(targets, outputs)
rmse = np.sqrt(mean_squared_error(outputs, targets))
mape = mean_absolute_percentage_error(outputs, targets)
mae = mean_absolute_error(outputs, targets)
np.save('test_result/result_{}.npy'.format(args.model_name_suffix), [outputs, targets])
print('Test Epoch: {} [{}/ \tLoss: {:.6f}, MAE: {:.2f}, RMSE :{:.3f}, R2: {:.2f}, MAPE: {:.3f}'.format(epoch, len(test_tiles), running_loss/len(test_tiles), mae, rmse, r_2, mape))
writer.add_scalar('test_loss', loss/len(test_tiles))
writer.add_scalar('test_mae', mae)
writer.add_scalar('test_rmse', rmse)
writer.add_scalar('test_mape', mape)
writer.add_scalar('test_R2', r_2)
writer.close()
