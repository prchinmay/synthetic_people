"""
Various important parameters of our model and training procedure.
"""
import numpy as np

def get_general_params():
    param = {}
    dn = 1
    param['IMG_HEIGHT'] = int(256/dn)
    param['IMG_WIDTH'] = int(256/dn)
    param['obj_scale_factor'] = 1.14/dn
    param['scale_max'] = 1.05  # Augmentation scaling
    param['scale_min'] = 0.90
    param['max_rotate_degree'] = 5
    param['max_sat_factor'] = 0.05
    param['max_px_shift'] = 10
    param['posemap_downsample'] = 2
    param['sigma_joint'] = 7/4.0
    param['n_joints'] = 14
    param['n_limbs'] = 10
    param['seed'] = 42
 
    #parent direvtory
    param['parent_dir'] = '/content/drive/MyDrive/surreal/play/for_report/'
    
    #import csv file paths
    param['pairs_list'] = param['parent_dir'] + '/pairs_list.csv'
    
    param['kp_src'] = param['parent_dir'] + '/kp/kp_src.csv'
    param['kp_tgt'] = param['parent_dir'] + '/kp/kp_tgt.csv'
    param['bbox_src'] = param['parent_dir'] + '/bbox/bbox_src.csv'
    param['bbox_tgt'] = param['parent_dir'] + '/bbox/bbox_tgt.csv'
    
    #param['results'] = param['parent_dir'] + "/pw_results"
    param['results'] = "/content/"
    param['char'] = param['results']  + 'char'
    param['mask'] = param['results']  + 'mask'

    

    # Using MPII-style joints: head (0), neck (1), r-shoulder (2), r-elbow (3), r-wrist (4), l-shoulder (5),
    # l-elbow (6), l-wrist (7), r-hip (8), r-knee (9), r-ankle (10), l-hip (11), l-knee (12), l-ankle (13)
    param['limbs'] = [[0, 1], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [2, 5, 8, 11]]
    param['plot'] = [[0, 1], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [1,2, 8, 11, 5,1]]
    param['kp_order'] = np.array([15,12,17,19,21,16,18,20,2,5,8,1,4,7])
    param['n_training_iter'] = 200000
    param['test_interval'] = 500
    param['model_save_interval'] = 1000
    
    param['weights'] = '/content/drive/MyDrive/posewarp-check/models/vgg+gan_5000.h5'
    #param['weights'] = '/content/drive/MyDrive/posewarp-check/models/vgg_100000.h5'
    #param['weights'] = '/content/drive/MyDrive/pw-train/models/surreal/500.h5'
    param['project_dir'] = '/content/drive/MyDrive/pw_new/'
    param['batch_size'] = 1
    return param

