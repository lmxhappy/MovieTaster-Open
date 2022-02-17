#!/usr/bin/env python
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

from tensorboard.plugins import projector

from tensorflow.keras.datasets.mnist import load_data

LOG_DIR = 'minimalsample'
NAME_TO_VISUALISE_VARIABLE = "mnistembedding"
TO_EMBED_COUNT = 500


path_for_mnist_sprites =  os.path.join(LOG_DIR,'mnistdigits.png')
path_for_mnist_metadata =  os.path.join(LOG_DIR,'metadata.tsv')

mnist = load_data()
batch_xs, batch_ys = mnist.train.next_batch(TO_EMBED_COUNT)



