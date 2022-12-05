# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 00:12:05 2022

@author: prchi
"""
import pandas as pd
import numpy as np
import json
import re


def reorder(row,indices):
    xs = row[1][0] + row[1][1:-1].strip(" ") + row[1][-1]
    xs = re.sub("\s+", ",", xs.strip())
    xs = np.array(json.loads(xs))
    xs = np.take(xs, indices)

    ys = row[2][0] + row[2][1:-1].strip(" ") + row[2][-1]
    ys = re.sub("\s+", ",", ys.strip())
    ys = np.array(json.loads(ys))
    ys = np.take(ys, indices)
    return xs,ys
    
def interp(xs, ys):
  #ys[0] = ys[0] + 3*(ys[0]-ys[1])
  #xs[0] = xs[0] + 3*(xs[0]-xs[1])
  
  xtmp = (xs[8] + xs[11])/2
  ytmp = (ys[8] + ys[11])/2
  
  xs[0] = xs[1] + (xs[1]-xtmp)/2
  ys[0] = ys[1] + (ys[1]-ytmp)/2
  return xs, ys

def clean(kp):
    kp = np.where(kp<0, -1, kp)
    arr = np.ptp(kp, axis=0)
    return kp,arr

def normalize(xs, ys):
    xs = xs/320
    ys = ys/240
    return xs,ys

def load_data(indices, path):
    
    df = pd.read_csv(path, header=None)
    cnt = len(df)

    train_data = []

    for i in range(cnt):
      #print('processing %d / %d ...' %(i, cnt))
      row = df.iloc[i]
      xs,ys = reorder(row,indices)
      kp = interp(xs,ys)
      #kp,arr =clean(kp)
      #xs,ys = normalize(xs,ys)
      kp = np.vstack((xs,ys))
      kp = kp.flatten()
      train_data.append(kp)

    train_data = np.array(train_data)
    df_data = pd.DataFrame(train_data)
    df_data['names'] = df.iloc[:][0]
    return df_data