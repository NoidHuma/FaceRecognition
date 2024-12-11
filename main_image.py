import numpy as np
import cv2
import warnings

warnings.filterwarnings('ignore')


def plot_image(img, emoj):
    wmin = 256
    hmin = 256

    emoj = cv2.resize(emoj, (wmin, hmin))
    img = cv2.resize(img, (wmin, hmin))
    cv2.imshow('Images', cv2.hconcat([img, emoj]))


def img_recognition(img, model, classes, face_cascade):
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    if len(faces) > 0:
        largest_face = max(faces, key=lambda face: face[2] * face[3])
    else:
        largest_face = None

    if largest_face is not None:
        x, y, w, h = largest_face

        gray = cv2.resize(gray[x:x + w, y:y + h], (48, 48))
        gray = np.expand_dims(gray, axis=-1)
        gray = np.expand_dims(gray, axis=0)

        pred = model.predict(gray)
        idx = pred.argmax(axis=-1)[0]

        emoj = cv2.imread(f'emojis/{classes[idx]}.jpg')

        plot_image(img, emoj)
    else:
        print("Лица не найдены")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
