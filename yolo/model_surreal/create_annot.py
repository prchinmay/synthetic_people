# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 02:22:29 2022

@author: prchi
"""
import pandas as pd
#from cfg import *
import cv2

#df_src = pd.read_csv(bbox_src)
#df_tgt = pd.read_csv(bbox_tgt)
#df_test = pd.read_csv(bbox_test)

#df_tgt = pd.concat([df_src, df_tgt])
#df_tgt = df_tgt.sample(frac=1, random_state=42).reset_index(drop=True)

#df_src = pd.read_csv('/content/drive/MyDrive/surreal/play/1000_intrp/bbox/bbox_src.csv')
#annot_bsl = '/content/drive/MyDrive/yolo/model_surreal/Train/baseline.txt'
df_src = pd.read_csv('/content/drive/MyDrive/surreal/play/1000_intrp/bbox/bbox_src.csv')
annot_bsl = '/content/aug_intrp.txt'
des_path = '/content/imgs'

def save_annot_train(des_path, df, path):
  #to strings
  df["xmin"] = df["xmin"].apply(str)
  df["ymin"] = df["ymin"].apply(str)
  df["xmax"] = df["xmax"].apply(str)
  df["ymax"] = df["ymax"].apply(str)
  df["label"] = (df["label"].apply(int)).apply(str)

  df['annot'] = df['names'] + " " + df[['xmin', 'ymin', 'xmax', 'ymax', 'label']].T.agg(','.join)
  df = df.drop(columns=['names','xmin', 'ymin', 'xmax', 'ymax', 'label'])

  for i in range(len(df)):
    annot = df.iloc[i]['annot']
    with open(path, 'a+') as fd:
      fd.write(annot)
      fd.write("\n")

def create_test_dir(df, dir_t):
#to strings
  df["xmin"] = df["xmin"].apply(str)
  df["ymin"] = df["ymin"].apply(str)
  df["xmax"] = df["xmax"].apply(str)
  df["ymax"] = df["ymax"].apply(str)
  df["label"] = (df["label"].apply(int)).apply(str)

  df['annot'] = df['names'] + " " + df[['xmin', 'ymin', 'xmax', 'ymax', 'label']].T.agg(','.join)
  df = df.drop(columns=['names','xmin', 'ymin', 'xmax', 'ymax', 'label'])

  for i in range(len(df)):
    print(i)
    annot = df.iloc[i]['annot']
    img_path = annot.split(" ")[0]
    img = cv2.imread(img_path)
    path = dir_t + "/" + img_path.split("/")[-1][:-4]

    #save img
    cv2.imwrite(path + ".jpg", img)

    #save annot
    with open(path + ".txt", 'w') as fd:
      fd.write(annot)
      fd.write("\n")
      

#save_annot_train(df_tgt, annot_aug)
save_annot_train(des_path, df_src, annot_bsl)
#create_test_dir(df_test, test_dir)