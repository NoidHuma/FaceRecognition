import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from PIL import Image

from sklearn import metrics

import cv2
import os

from glob import glob

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator

import warnings

warnings.filterwarnings('ignore')

path = 'dataset/train'
classes = os.listdir(path)
print(classes)

count = []
for cat in classes:
    count.append(len(os.listdir(f'{path}/{cat}')))
sb.barplot(x=classes, y=count)
plt.show()
