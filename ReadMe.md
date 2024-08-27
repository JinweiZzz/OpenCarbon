# OpenCarbon

This repo is the official implementation for the paper   
OpenCarbon: A Multi-level Neural Approach for High-Resolution Carbon Emission Prediction with Multi-modality Open Data

## Overall architecture
This work aims to construct a prediction framework that predicts high-resolution carbon emissions with open data of satellite images and POI. 
![Overall framework](framework.pdf)


## Data
We conduct experiments on cities that span both developed and developing countries, including New York, London, and Beijing. Summary of the datasets are presented:

| Region         | Area      | POI Source   | Target Year | #POI Categories |
| -------------- | --------- | ------------ | ----------- | --------------- |
| Beijing        | 1381 km²  | Tencent Map  | 2018        | 14              |
| Great London   | 778 km²   | SafeGraph    | 2018        | 14              |
| New York City  | 1569 km²  | SafeGraph    | 2019        | 15              |

*Table 1: The summary statistics of our datasets.*

Due to the size limit of github, we have stored the data in an anonymous google drive link: https://drive.google.com/drive/folders/1_HHa5X6nLiB4mHfEIn42jb5fwc64nf0v?usp=sharing. [Due to cloud storage limit, we only update the Beijing dataset. All other datasets are available through email requests.] Please download them and place them inside /data.

## Installation
### Environment
- Tested OS: Linux
- Python >= 3.8
- torch == 2.2.1
- Tensorboard


## Config 
Configs for performance reproductions on all datasets. 


### Beijing
```
python main.py --lr 5e-5 --batch_size 32 --city beijing --neighbor_size 3
```

### New York
```
python main.py --lr 1e-5 --batch_size 32 --city london --neighbor_size 3
```

### London
```
python main.py --lr 5e-5 --batch_size 16 --city newyork --neighbor_size 5
```
