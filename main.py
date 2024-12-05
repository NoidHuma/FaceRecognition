import numpy as np

import cv2

import os

from keras.models import load_model

import warnings

warnings.filterwarnings('ignore')


model = load_model('my_model.keras')

classes = os.listdir("dataset/train")


def plot_image(img, emoj):
    wmin = 256
    hmin = 256

    emoj = cv2.resize(emoj, (wmin, hmin))
    img = cv2.resize(img, (wmin, hmin))
    cv2.imshow('Images', cv2.hconcat([img, emoj]))
