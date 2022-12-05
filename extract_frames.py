#Define Paths
video_path = "/content/drive/MyDrive/surreal/data/cmu/train/run0"
rgb_path = "/content/drive/MyDrive/surreal/train/frames"
bbox_file = "/content/drive/MyDrive/surreal/train/bbox/bbox.csv"
kp_file = "/content/drive/MyDrive/surreal/train/keypoints/kp_surreal.csv"
mask_path = "/content/drive/MyDrive/surreal/train/masks"


#Define Variables
n_frames = 4
#n_video = 100

#imports
import numpy as np
import glob
import os
#from matplotlib import pyplot as plt
import cv2
import scipy.io
#import random
#import csv   
import pandas as pd

list_run = os.listdir(video_path)

def get_frame(vid, seg, frame_no):
    vid.set(1, int(frame_no))
    ret, still = vid.read()
    mask = seg["segm_" + str(frame_no+1)]
    mask = 1-(mask<1)
    return still, mask

   
def get_bbox(mask):
    mask = mask.astype(np.uint8)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    x,y,w,h = cv2.boundingRect(contours[0])
    return [x,y,x+w,y+h]

def save_person(frame, out_name):
    cv2.imwrite(out_name, frame)

def save_kp(info,frame_no,out_name):
    kp = info["joints2D"]
    kp = kp[:,:,frame_no]
    kp = kp.astype(int) 
    data = {'file': out_name,'xs': [kp[0]], 'ys': [kp[1]]}
    df = pd.DataFrame(data)
    df.to_csv(kp_file, mode='a', index=False, header=None)
    
def save_mask(mask, out_name, mask_path):
    path = os.path.join(mask_path, out_name.rsplit("/")[-1])
    mask = mask*255
    cv2.imwrite(path, mask)


def save_bbox(bbox, bbox_file, out_name):
    xmin,ymin,xmax,ymax = str(bbox[0]),str(bbox[1]),str(bbox[2]),str(bbox[3])
    annot = out_name + "," + xmin + "," + ymin + "," + xmax + "," + ymax + "," + str(0)
    with open(bbox_file, 'a') as fd:
        fd.write(f'\n{annot}')

#lst = glob.glob(video_path+"/**/*.mp4", recursive=True)
#print(len(lst))

for i, video in enumerate(glob.iglob(video_path+"/**/*.mp4", recursive=True)):
    print(i)
    vid = cv2.VideoCapture(video)
    seg = scipy.io.loadmat(video[:-4] + "_segm.mat")
    info = scipy.io.loadmat(video[:-4] + "_info.mat")
    total_frames = vid.get(7)
    print(vid)
    for i in range(0, n_frames):
        frame_no = int(i*total_frames/n_frames)
        frame, mask = get_frame(vid,seg,frame_no)
        percentage = np.sum(mask)/(frame.shape[0]*frame.shape[1])
        if percentage < 0.01:
            #print(percentage*100)
            continue

        #bbox = get_bbox(mask)
        out_name = '{}_{}.jpg'.format(os.path.join(rgb_path,video.rsplit("/")[-1])[:-4], str(frame_no+1))

        save_bbox(bbox, bbox_file, out_name)
        save_kp(info,frame_no,out_name)
        save_person(frame,out_name)
        save_mask(mask, out_name, mask_path)

        