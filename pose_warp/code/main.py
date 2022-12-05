
import data_generation_scaling
#import data_generation
import networks
import param
from util import viz_in_out, transform, save_output, save_mask
import pandas as pd 
import numpy as np

def test():

    params = param.get_general_params()
    pairs_list = pd.read_csv(params['pairs_list'])
    
    test_feed = data_generation_scaling.create_feed(params, pairs_list)
    
    generator = networks.network_posewarp(params)
    generator.load_weights(params['weights'])

    for i in range(len(pairs_list)):
      print(i)
      x = next(test_feed)
      gen = generator.predict(x[:5])
      
      save_mask(gen, pairs_list.iloc[i], params)
      
      #viz(src_img, out_img, tgt_pose, params, path)
      #src_img, out_img, out_per, tgt_pose = transform(x, gen)

if __name__ == "__main__":
    test()