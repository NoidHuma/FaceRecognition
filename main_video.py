import numpy as np
import cv2
import warnings

warnings.filterwarnings('ignore')


# Функция для отображения изображения и эмодзи
def plot_image(img, emoj):
    wmin = 256
    hmin = 256

    emoj = cv2.resize(emoj, (wmin, hmin))
    img = cv2.resize(img, (wmin, hmin))
    cv2.imshow('Images', cv2.hconcat([img, emoj]))


def video_recognition(model, classes, face_cascade):
    # Захват видео с веб-камеры
    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()
        if not ret:
            break  # Если не удалось захватить кадр, выйти из цикла

        # Изменение размера изображения для обработки
        img = cv2.resize(img, (256, 256))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Обнаружение лиц на изображении
        faces = face_cascade.detectMultiScale(gray)

        if len(faces) > 0:
            # Нахождение самого большого лица
            largest_face = max(faces, key=lambda face: face[2] * face[3])
            x, y, w, h = largest_face

            # Вырезание области лица и изменение размера для предсказания
            gray_face = cv2.resize(gray[y - 5:y + h + 15, x + 4:x + w - 4], (48, 48))
            gray_face = np.expand_dims(gray_face, axis=-1)
            gray_face = np.expand_dims(gray_face, axis=0)

            # Предсказание эмоции
            pred = model.predict(gray_face)
            idx = pred.argmax(axis=-1)[0]

            # Загружаем соответствующий эмодзи
            emoj = cv2.imread(f'emojis/{classes[idx]}.jpg')
            plot_image(img, emoj)
        else:
            # Если лицо не найдено, показываем изображение "Нет лица"
            emoj = cv2.imread('images/NofaceDetected.png')
            plot_image(img, emoj)

        # Условие выхода из цикла
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()
