import os
import numpy as np
import cv2
import transformations
import scipy.io as sio
import glob
import pandas as pd
import json
import re


def get_person_scale(joints):
    upper_body_size = (-joints[0][1] + (joints[8][1] + joints[11][1]) / 2.0)
    rcalf_size = np.sqrt((joints[9][1] - joints[10][1]) ** 2 + (joints[9][0] - joints[10][0]) ** 2)
    lcalf_size = np.sqrt((joints[12][1] - joints[13][1]) ** 2 + (joints[12][0] - joints[13][0]) ** 2)
    calf_size = (lcalf_size + rcalf_size) / 2.0

    size = np.max([2.5 * upper_body_size, 5.0 * calf_size])
    return size / 200.0


def read_frame(kp,bbox):

    joints = (kp.to_numpy()).reshape(2,14)
    joints = joints.T
    joints = joints - 1.0

    #bbox
    xmin = bbox['xmin']
    ymin = bbox['ymin']
    xmax = bbox['xmax']
    ymax = bbox['ymax']
    box_frame = [xmin, ymin, xmax-xmin, ymax-ymin]

    scale = get_person_scale(joints)
    pos = np.zeros(2)
    pos[0] = (box_frame[0] + box_frame[2] / 2.0)
    pos[1] = (box_frame[1] + box_frame[3] / 2.0)

    return joints, scale, pos


def warp_example_generator(param, pairs_list, do_augment=False, return_pose_vectors=True):
    img_width = param['IMG_WIDTH']
    img_height = param['IMG_HEIGHT']
    pose_dn = param['posemap_downsample']
    sigma_joint = param['sigma_joint']
    n_joints = param['n_joints']
    scale_factor = param['obj_scale_factor']
    batch_size = param['batch_size']
    limbs = param['limbs']
    n_limbs = param['n_limbs']
    
    #file imports
    kps_src = pd.read_csv(param['kp_src'])
    bboxs_src = pd.read_csv(param['bbox_src'])
    kps_tgt = pd.read_csv(param['kp_tgt'])
    bboxs_tgt = pd.read_csv(param['bbox_tgt'])
    
    #count
    i = 0

    while True:
        x_src = np.zeros((batch_size, img_height, img_width, 3))
        x_mask_src = np.zeros((batch_size, img_height, img_width, n_limbs + 1))
        x_pose_src = np.zeros((batch_size, int(img_height / pose_dn), int(img_width / pose_dn), n_joints))
        x_pose_tgt = np.zeros((batch_size, int(img_height / pose_dn), int(img_width / pose_dn), n_joints))
        x_trans = np.zeros((batch_size, 2, 3, n_limbs + 1))
        x_posevec_src = np.zeros((batch_size, n_joints * 2))
        x_posevec_tgt = np.zeros((batch_size, n_joints * 2))
        
        #select pairs
        pair = pairs_list.iloc[i]

        #select keypoints corresponsing to names in pairs list
        bbox_src = bboxs_src.loc[bboxs_src['names']==pair['names_src']].drop(columns = ["names"])
        bbox_tgt = bboxs_tgt.loc[bboxs_tgt['names']==pair['names_tgt']].drop(columns = ["names"])
    
        
        #select bboxes corresponsing to names in pairs list
        kp_src = kps_src.loc[kps_src['names'].str.split('/',expand=True)[7]==pair['names_src'].rsplit('/')[-1]].drop(columns = ["names"])
        kp_tgt = kps_tgt.loc[kps_tgt['names']==pair['names_tgt']].drop(columns = ["names"])

        I0 = cv2.imread(pair['names_src'])
        
        joints0, scale0, pos0 = read_frame(kp_src,bbox_src)
        joints1, scale1, pos1 = read_frame(kp_tgt,bbox_tgt)

        if scale0 > scale1:
            scale = scale_factor / scale0
        else:
            scale = scale_factor / scale1

        pos = (pos0 + pos1) / 2.0
        
        I0, joints0 = center_and_scale_image(img_width, img_height, pos, scale, joints0, I=I0, image = True)
        joints1 = center_and_scale_image(img_width, img_height, pos, scale, joints1, I=None, image = False)

        I0 = (I0 / 255.0 - 0.5) * 2.0

        posemap0 = make_joint_heatmaps(img_height, img_width, joints0, sigma_joint, pose_dn)
        posemap1 = make_joint_heatmaps(img_height, img_width, joints1, sigma_joint, pose_dn)

        src_limb_masks = make_limb_masks(limbs, joints0, img_width, img_height)
        src_bg_mask = np.expand_dims(1.0 - np.amax(src_limb_masks, axis=2), 2)
        src_masks = np.log(np.concatenate((src_bg_mask, src_limb_masks), axis=2) + 1e-10)

        x_src[:,:, :, :] = I0
        x_pose_src[:,:, :, :] = posemap0
        x_pose_tgt[:,:, :, :] = posemap1
        x_mask_src[:,:, :, :] = src_masks
        x_trans[:,:, :, 0] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        x_trans[:,:, :, 1:] = get_limb_transforms(limbs, joints0, joints1)

        x_posevec_src[:,:] = joints0.flatten()
        x_posevec_tgt[:,:] = joints1.flatten()

        #output organizing
        out = [x_src, x_pose_src, x_pose_tgt, x_mask_src, x_trans]
       
        if return_pose_vectors:
            out.append(x_posevec_src)
            out.append(x_posevec_tgt)
            
        i = i+1   
        
        yield (out)


def create_feed(params, pairs_list, return_pose_vectors=True, transfer=False):
        
    feed = warp_example_generator(params, pairs_list, return_pose_vectors=True)

    return feed

def center_and_scale_image(img_width, img_height, pos, scale, joints, I= None, image = False):
    joints = joints * scale
    x_offset = (img_width - 1.0) / 2.0 - pos[0] * scale
    y_offset = (img_height - 1.0) / 2.0 - pos[1] * scale
    joints[:, 0] += x_offset
    joints[:, 1] += y_offset
    
    if image:
        I = cv2.resize(I, (0, 0), fx=scale, fy=scale)
        T = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
        I = cv2.warpAffine(I, T, (img_width, img_height))
        return I, joints
    
    else:
        return joints
    

def make_joint_heatmaps(height, width, joints, sigma, pose_dn):
    height = int(height / pose_dn)
    width = int(width / pose_dn)
    n_joints = joints.shape[0]
    var = sigma ** 2
    joints = joints / pose_dn

    H = np.zeros((height, width, n_joints))

    for i in range(n_joints):
        if (joints[i, 0] <= 0 or joints[i, 1] <= 0 or joints[i, 0] >= width - 1 or
                joints[i, 1] >= height - 1):
            continue

        H[:, :, i] = make_gaussian_map(width, height, joints[i, :], var, var, 0.0)

    return H


def make_gaussian_map(img_width, img_height, center, var_x, var_y, theta):
    xv, yv = np.meshgrid(np.array(range(img_width)), np.array(range(img_height)),
                         sparse=False, indexing='xy')

    a = np.cos(theta) ** 2 / (2 * var_x) + np.sin(theta) ** 2 / (2 * var_y)
    b = -np.sin(2 * theta) / (4 * var_x) + np.sin(2 * theta) / (4 * var_y)
    c = np.sin(theta) ** 2 / (2 * var_x) + np.cos(theta) ** 2 / (2 * var_y)

    return np.exp(-(a * (xv - center[0]) * (xv - center[0]) +
                    2 * b * (xv - center[0]) * (yv - center[1]) +
                    c * (yv - center[1]) * (yv - center[1])))


def make_limb_masks(limbs, joints, img_width, img_height):
    n_limbs = len(limbs)
    mask = np.zeros((img_height, img_width, n_limbs))

    # Gaussian sigma perpendicular to the limb axis.
    sigma_perp = np.array([11, 11, 11, 11, 11, 11, 11, 11, 11, 13]) ** 2

    for i in range(n_limbs):
        n_joints_for_limb = len(limbs[i])
        p = np.zeros((n_joints_for_limb, 2))

        for j in range(n_joints_for_limb):
            p[j, :] = [joints[limbs[i][j], 0], joints[limbs[i][j], 1]]

        if n_joints_for_limb == 4:
            p_top = np.mean(p[0:2, :], axis=0)
            p_bot = np.mean(p[2:4, :], axis=0)
            p = np.vstack((p_top, p_bot))

        center = np.mean(p, axis=0)

        sigma_parallel = np.max([5, (np.sum((p[1, :] - p[0, :]) ** 2)) / 1.5])
        theta = np.arctan2(p[1, 1] - p[0, 1], p[0, 0] - p[1, 0])

        mask_i = make_gaussian_map(img_width, img_height, center, sigma_parallel, sigma_perp[i], theta)
        mask[:, :, i] = mask_i / (np.amax(mask_i) + 1e-6)

    return mask


def get_limb_transforms(limbs, joints1, joints2):
    n_limbs = len(limbs)

    Ms = np.zeros((2, 3, n_limbs))

    for i in range(n_limbs):
        n_joints_for_limb = len(limbs[i])
        p1 = np.zeros((n_joints_for_limb, 2))
        p2 = np.zeros((n_joints_for_limb, 2))

        for j in range(n_joints_for_limb):
            p1[j, :] = [joints1[limbs[i][j], 0], joints1[limbs[i][j], 1]]
            p2[j, :] = [joints2[limbs[i][j], 0], joints2[limbs[i][j], 1]]

        tform = transformations.make_similarity(p2, p1, False)
        Ms[:, :, i] = np.array([[tform[1], -tform[3], tform[0]], [tform[3], tform[1], tform[2]]])

    return Ms

#def save_bbox()