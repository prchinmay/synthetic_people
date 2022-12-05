# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 23:33:24 2022

@author: prchi
"""

import numpy as np
def get_general_params():

    param = {}

    #parent csv files
    param['kp_parent'] = "/content/drive/MyDrive/surreal/train/keypoints/kp_surreal.csv"
    param['bbox_parent'] = "/content/drive/MyDrive/surreal/train/bbox/bbox.csv"
    #param['kp_parent'] = "/content/drive/MyDrive/surreal/demo/test/keypoints/kp_surreal.csv"
    #param['bbox_parent'] = "/content/drive/MyDrive/surreal/demo/test/bbox/bbox.csv"

    #main dir
    param['main_dir'] = '/content/drive/MyDrive/surreal/play/1000_open'
    
    ###----------------MATCH THESE PATHS WITH POSWARP INPUT PATHS-----------------------
    #src_csv files
    param['kp_src'] = param['main_dir'] + "/kp/kp_src_25k.csv"
    param['bbox_src'] = param['main_dir'] + "/bbox/bbox_src_25k.csv"
    #param['bbox_src'] = param['main_dir'] + "/bbox/bbox_src_2k.csv"
    
    #tgt_csv files
    param['kp_tgt'] = param['main_dir'] + "/kp/kp_tgt.csv"
    param['bbox_tgt'] = param['main_dir'] + "/bbox/bbox_tgt.csv"
    param['bbox_test'] = param['main_dir'] + "/bbox/bbox_test.csv"
    
    # posewarp Input/Output
    param['pairs_list'] = param['main_dir'] + '/pairs_list.csv'
    ###-----------------------------------------------------------------------------------
    
    # pw output and keypoint plot directories
    param['plot_kp_src_dir'] = param['main_dir'] + "/kp_plots/src"
    param['plot_kp_tgt_dir'] = param['main_dir'] + "/kp_plots/tgt"
    
    #other plots
    param['plot_n_clusters'] = param['main_dir'] + "/other_plots/aic_vs_clusters.jpg"
    param['tsne_2d_src'] =  param['main_dir'] +"/other_plots/tsne_2d_src.jpg"
    param['tsne_2d_tgt'] =  param['main_dir'] + "/other_plots/tsne_2d_tgt.jpg"

       
    #variables
    param['seed'] = 42
    param['n_src'] = 25000 
    param['n_tgt'] = 4000
    param['n_test'] = 200
    param['maximum'] = 500
    param['step'] = 50
    param['img_width'] = 320
    param['img_height'] = 240

    #constants
    param['image_size'] = (320,240)
    param['indices'] = np.array([15,12,17,19,21,16,18,20,2,5,8,1,4,7])
    param['order'] = [[0, 1], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [1,2, 8, 11, 5,1]]
    
    return param