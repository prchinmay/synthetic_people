# -*- coding: utf-8 -*-

"""
Created on Sun Jul 31 00:03:48 2022

@author: prchi
"""

import pandas as pd
import numpy as np
from sklearn import mixture
import params
from utils import clip, n_clusters, kp2bbox, bbox2cnt
from saves import create_bbox_src, create_pairs
from dataloader import load_data
import os

param = params.get_general_params()

# Check whether the specified path exists or not
isExist = os.path.exists(param['main_dir'])

if not isExist:
  os.makedirs(param['main_dir'])
  os.makedirs(param['main_dir'] + "/" + "kp")
  os.makedirs(param['main_dir'] + "/" + "bbox")
  os.makedirs(param['main_dir'] + "/" + "other_plots")
  os.makedirs(param['main_dir'] + "/" + "pw_results")
  print("folders created")

seed = param['seed']
n_src = param['n_src']
n_tgt = param['n_tgt']
n_test = param['n_test']
img_width = param['img_width']
img_height = param['img_height']
indices = param['indices']
np.random.seed(seed) 

"""
Start
"""

##load data
train_data = load_data(indices, param['kp_parent'])
train_data = (train_data.sample(n_src, random_state = seed)).reset_index(drop=True)


##select only some
df_kp_src = train_data[:n_src]


##quick saves src
df_bbox_src = create_bbox_src(df_kp_src, param['bbox_parent'])
df_kp_src.to_csv(param['kp_src'], index=False)
df_bbox_src.to_csv(param['bbox_src'], index = False)


