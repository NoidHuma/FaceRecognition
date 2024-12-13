import numpy as np
import cv2
import warnings
from PIL import Image, ImageTk


warnings.filterwarnings('ignore')


def plot_image(app, result_name):
    img = app.img.resize((256, 256))
    emoj = Image.open(f'emojis/{result_name}.jpg').resize((256, 256))
    app.img_photo = ImageTk.PhotoImage(img)
    app.emoj_photo = ImageTk.PhotoImage(emoj)
    app.img_label.config(image=app.img_photo)
    app.emoj_label.config(image=app.emoj_photo)


def img_recognition(app):

    gray = cv2.cvtColor(np.array(app.img), cv2.COLOR_BGR2GRAY)
    faces = app.face_cascade.detectMultiScale(gray)

    if len(faces) > 0:
        largest_face = max(faces, key=lambda face: face[2] * face[3])
    else:
        largest_face = None

    if largest_face is not None:
        x, y, w, h = largest_face

        gray = cv2.resize(gray[x:x + w, y:y + h], (48, 48))
        gray = np.expand_dims(gray, axis=-1)
        gray = np.expand_dims(gray, axis=0)

        pred = app.model.predict(gray)
        idx = pred.argmax(axis=-1)[0]

        result_name = app.classes[idx]

    else:
        result_name = "NofaceDetected"

    plot_image(app, result_name)


