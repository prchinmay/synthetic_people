import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
#import pandas as pd
from PIL import Image

def vgg_preprocess(x):
	x = 255.0 * (x + 1.0)/2.0

	x[:,:,:,0] -= 103.939
	x[:,:,:,1] -= 116.779
	x[:,:,:,2] -= 123.68

	return x


def viz_in_out(src_img, out_img, tgt_pose, params, path):
    joints = params['plot']
    xs = tgt_pose[:,0]
    ys = tgt_pose[:,1]

    ones = np.ones((256,256,3))
    plt.subplot(2,3,2)
    for j in joints:
      plt.plot(np.take(xs, j),np.take(ys, j),"-o")
    plt.imshow(ones)

    plt.subplot(2,3,4)
    plt.imshow(src_img)

    plt.subplot(2,3,5)
    plt.imshow(out_img)
    
    plt.savefig(path)
    plt.clf()
    plt.cla()

def save_output(out_img, path):
    out_img = Image.fromarray(out_img)
    out_img.save(path)
    
def read_sets(path):
    file = open(path, "r")
    output = file.read()
    output = output.rsplit("\n")
    return output

def aug(x, gen, path):
    out = x[0][0]
    for i in range(len(gen[0])):
        #out = (x_src * bg_mask) + fg_tgt
        out = np.multiply(out,gen[2][i]) + np.array(gen[1][i])
        
    out = ((out/2)+0.5)*255
    out = out.astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    out = Image.fromarray(out)
    out.save(path)
    return out


def transform(x, gen):
    #src_img = ((np.array(x[0][0])/2)+0.5)*255
    #src_img = src_img.astype(np.uint8)
    #tgt_pose = np.array(x[6][0]) 
    out_per = ((gen[1][0]/2)+0.5)*255
    #out_img = ((gen[0][0]/2)+0.5)*255
    #out_img = out_img.astype(np.uint8)
    #out_per = out_per.astype(np.uint8)
    
    #src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    #out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    out_per = cv2.cvtColor(out_per, cv2.COLOR_BGR2RGB)
    #tgt_pose = tgt_pose.reshape(14,2)
    
    #return src_img, out_img, out_per, tgt_pose
    return out_per

def save_mask(gen, names, param):
    fname = str(names['names_src']).rsplit("/")[-1][:-4] + "_+_" + str(names['names_tgt']) + '.jpg'
    path_char = param['char'] + "/"+ fname 
    path_mask = param['mask'] + "/" + fname

    
    char = np.array(gen[1][0])
    mask = np.array(gen[2][0])
    char = ((char/2)+0.5)*255
   
    
    cv2.imwrite(path_char, char)
    np.save(path_mask, mask) # save
 
            
    