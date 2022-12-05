# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 23:34:49 2022

@author: prchi
"""

import pandas as pd
import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt

def bbox2cnt(df):
    dfn = pd.DataFrame()
    dfn['x_cnt'] = df['xmin'] + (df['xmax'] - df['xmin'])/2
    dfn['y_cnt'] = df['ymin'] + (df['ymax'] - df['ymin'])/2
    return dfn

def resize(kp, des_w, des_h, img_width, img_height):
    kpn = np.zeros((len(kp), 28))
    scale_x = des_w/img_width
    scale_y =  des_h/img_height
    kpn[:,:14] = kp[:,:14]*scale_x
    kpn[:,14:28] = kp[:,14:28]*scale_y
    return kpn

def clip(dfn, xmax, ymax):
    #[-4, 4,-5,7]
    dfn["xmin"] =  (dfn["xmin"].to_numpy()).clip(min=0) 
    dfn["xmax"] =  (dfn["xmax"].to_numpy()).clip(max=xmax)
    dfn["ymin"] =  (dfn["ymin"].to_numpy()).clip(min=0) 
    dfn["ymax"] =  (dfn["ymax"].to_numpy()).clip(max=ymax) 
    return dfn

def kp2bbox(kp):
    ar = np.array(kp)
    dfn = pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax'])
    dfn["xmin"] = np.min(ar[:,:14], axis=1)
    dfn["ymin"] = np.min(ar[:,14:28], axis=1)
    dfn["xmax"] = np.max(ar[:,:14], axis=1)
    dfn["ymax"] = np.max(ar[:,14:28], axis=1)
    return dfn

def n_clusters(X,param,seed):
    maximum = param['maximum']
    step = param['step']
    path = param['plot_n_clusters']
    
    n_components = np.arange(1, maximum, step)
    models = [mixture.GaussianMixture(n, covariance_type='full', random_state=seed) for n in n_components]
    aics = [model.fit(X).aic(X) for model in models]
    plt.plot(n_components, aics)
    plt.savefig(path)
    return n_components[np.argmin(aics)]


        
