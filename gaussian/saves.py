# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 23:58:32 2022

@author: prchi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib as mpl

def create_bbox_src(df_src, bbox_parent):
    df_p = pd.read_csv(bbox_parent, header=None)
    df = df_p.loc[df_p[0].isin(df_src['names'])]
    df = df.rename(columns={0: "names", 1: "xmin", 2: "ymin", 3: "xmax", 4: "ymax", 5: "label"})
    return df
      
def create_pairs(df_bbox_src, n_tgt):
    names = list(map(str, range(n_tgt)))
    df = pd.DataFrame(columns=['names_src','names_tgt'])
    #df['names_src'] = df_bbox_src.sample(n=n_tgt, replace=True, random_state=param['seed'])['names']
    df['names_src'] = pd.concat([df_bbox_src['names']]*4)
    df['names_tgt'] =  names  
    df = df.reset_index(drop=True)
    return df
    
def save_sets(df_pairs, path):
  unique_names = df_pairs['names_src'].unique()
  for i in unique_names:
    tgts = df_pairs['names_tgt'][df_pairs['names_src']==i].values[:]
    annot = i + ":" + str(list(tgts))
    with open(path, 'a+') as fd:
      fd.write(annot)
      fd.write("\n")

def plot_kp(kp, param, sdir, df):
    joints = param['order']
    ones = np.zeros((240,320,3))
    
    for i in range(len(kp)):
        row = kp[i]
        row = row.reshape(2,14)
        xs = row[0,:]
        ys = row[1,:]

        for j in joints:
            plt.plot(np.take(xs, j),np.take(ys, j),"-o")
        plt.imshow(ones.astype(np.uint8))
        
def plot_tsne_2d(X, labels, n_cluster, path):
    tsne = TSNE(n_components=2, perplexity=30)
    x_tsne = tsne.fit_transform(X)
    #fig, ax = plt.subplots(1,1, figsize=(10,6))
    N = n_cluster
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.linspace(0, N, N+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # make the scatter
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=labels, s=10, cmap=cmap, norm=norm)
    plt.show()
    plt.savefig(path)
    plt.clf()
    plt.cla()