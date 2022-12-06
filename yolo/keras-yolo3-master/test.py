import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from tensorflow.python.platform import build_info as build

if tf.test.gpu_device_name(): 
   print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
   print("No GPU Devices Found!")

print('Python version {}'.format(sys.version))
print('Tensorflow version: {}'.format(tf.__version__))
print('Keras version: {}'.format(keras.__version__))
#print('Cudnn version: {}'.format(build.build_info['cudnn_version']))
#print('Cuda version: {}'.format(build.build_info['cuda_version']))
print('GPU Available: {}'.format(len(tf.compat.v1.config.experimental.list_physical_devices('GPU'))))
