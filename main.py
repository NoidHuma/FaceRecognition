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


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('images/sad4.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray)

if len(faces) > 0:
    largest_face = max(faces, key=lambda face: face[2] * face[3])
else:
    largest_face = None
