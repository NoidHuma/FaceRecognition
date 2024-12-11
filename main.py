import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import main_image  # Импортируем модуль с функцией img_recognition
import main_video  # Импортируем модуль с функцией video_recognition
from keras.models import load_model


class EmotionRecognitionApp:
    model = load_model('my_model.h5')
    classes = os.listdir("dataset_mini/train")
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Recognition App")

        self.img = None  # Переменная для хранения выбранного изображения

        # Кнопка "Выбрать изображение"
        self.select_button = tk.Button(root, text="Выбрать изображение", command=self.select_image)
        self.select_button.pack(pady=10)

        # Метка для статуса изображения
        self.status_label = tk.Label(root, text="Изображение не выбрано", fg="red")
        self.status_label.pack(pady=10)

        # Кнопка "Распознать эмоцию на изображении"
        self.recognize_button = tk.Button(root, text="Распознать эмоцию на изображении", command=self.recognize_emotion,
                                          state=tk.DISABLED)
        self.recognize_button.pack(pady=10)

        # Кнопка "Распознавать эмоции с видеопотока"
        self.video_button = tk.Button(root, text="Распознавать эмоции с видеопотока", command=self.video_recognition)
        self.video_button.pack(pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(title="Выберите изображение",
                                               filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.img = file_path  # Сохраняем путь к изображению
            self.status_label.config(text="Изображение выбрано", fg="green")
            self.recognize_button.config(state=tk.NORMAL)  # Активируем кнопку распознавания

    def recognize_emotion(self):
        if self.img:
            main_image.img_recognition(self.img, self.model, self.classes, self.face_cascade)  # Вызываем функцию распознавания эмоции
        else:
            messagebox.showerror("Ошибка", "Пожалуйста, выберите изображение.")

    def video_recognition(self):
        main_video.video_recognition(self.model, self.classes, self.face_cascade)  # Вызываем функцию распознавания эмоций с видеопотока


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.mainloop()
