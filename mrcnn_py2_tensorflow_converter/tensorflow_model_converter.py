import numpy as np
import os
import sys
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib,utils

from keras import backend as K
import tensorflow as tf

class BalloonConfig(Config):

    NAME = "daily_objects" # your dataset name 
    NUM_CLASSES = 1+3 #your num of classes
    DETECTION_MIN_CONFIDENCE = 0.8
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

if __name__ == '__main__':

    import argparse
    
    weights_path = "directory to you trained h5 file.h5" # you need to modify here

    config = BalloonConfig()
    config.display()

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir="./logtest")
    model.load_weights(weights_path, by_name=True)

    sess = K.get_session()
    saver = tf.train.Saver()
    saver.save(sess,"./log/converted") # you may want to modify here

    

