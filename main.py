import os
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import main_image  # Импортируем модуль с функцией img_recognition
from keras.models import load_model


class EmotionRecognitionApp:

    def __init__(self, root):
        self.model = load_model('my_model.h5')
        self.classes = os.listdir("dataset_mini/train")
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        self.root = root
        self.root.title("Распознавание эмоций")
        w = 800
        h = 720
        x = (self.root.winfo_screenwidth() - w) / 2
        y = (self.root.winfo_screenheight() - h) / 2 - 50
        self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.root["bg"] = "LightYellow"

        self.img = None

        self.img_photo = None  # Переменная для хранения выбранного изображения
        self.emoj_photo = None

        self.video_capture = None
        self.is_video_running = False

        # Кнопка "Выбрать изображение"
        self.select_button = tk.Button(root,
                                       text="Выбрать изображение",
                                       command=self.select_image,
                                       bg="Goldenrod",
                                       activebackground="Khaki",
                                       font=("Arial", 24, "bold"),
                                       fg="White",
                                       activeforeground="White",
                                       disabledforeground="LightGoldenrodYellow",
                                       borderwidth=3,
                                       relief="solid"
                                       )
        self.select_button.pack(pady=[30, 0])

        # Метка для статуса изображения
        self.status_label = tk.Label(root,
                                     text="Изображение не выбрано",
                                     bg="LightYellow",
                                     font=("Arial", 16, "normal"),
                                     fg="red",
                                     )
        self.status_label.pack(pady=10)

        # Кнопка "Распознать эмоцию на изображении"
        self.recognize_button = tk.Button(root,
                                          text="Распознать эмоцию на изображении",
                                          command=self.recognize_emotion,
                                          bg="Goldenrod",
                                          activebackground="Khaki",
                                          font=("Arial", 24, "bold"),
                                          fg="White",
                                          activeforeground="White",
                                          disabledforeground="LightGoldenrodYellow",
                                          borderwidth=3,
                                          relief="solid",

                                          state=tk.DISABLED)
        self.recognize_button.pack(pady=30)

        # Кнопка "Распознавать эмоции с видеопотока"
        self.video_button = tk.Button(root,
                                      text="Распознавать эмоции с видеопотока",
                                      command=self.start_video,
                                      bg="Goldenrod",
                                      activebackground="Khaki",
                                      font=("Arial", 24, "bold"),
                                      fg="White",
                                      activeforeground="White",
                                      borderwidth=3,
                                      relief="solid"
                                      )
        self.video_button.pack(pady=30)

        # Frame для размещения img_label и emoj_label
        self.image_frame = tk.Frame(root,
                                    bg="LightYellow",
                                    borderwidth=2,
                                    relief="solid")
        self.image_frame.pack(pady=10)

        # Метка для изображения
        self.img_label = tk.Label(self.image_frame,
                                  image=ImageTk.PhotoImage(Image.open("images/white.jpg")),
                                  width=256, height=256
                                  )
        self.img_label.pack(side=tk.LEFT)

        # Метка для изображения эмоции
        self.emoj_label = tk.Label(self.image_frame,
                                   image=ImageTk.PhotoImage(Image.open("images/white.jpg")),
                                   width=256, height=256,
                                   )
        self.emoj_label.pack(side=tk.LEFT)

    def select_image(self):
        file_path = filedialog.askopenfilename(title="Выберите изображение",
                                               filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.img = Image.open(file_path)  # Сохраняем путь к изображению
            self.status_label.config(text="Изображение выбрано", fg="green")
            self.recognize_button.config(state=tk.NORMAL)  # Активируем кнопку распознавания

    def recognize_emotion(self):
        main_image.img_recognition(self)  # Вызываем функцию распознавания эмоции

    def start_video(self):
        self.status_label.config(text="Изображение не выбрано",
                                 fg="red")
        self.recognize_button.config(state=tk.DISABLED)
        self.select_button.config(state=tk.DISABLED)
        self.cap = cv2.VideoCapture(0)  # Открываем вебкамеру
        self.video_button.config(text="Прекратить распознавание",
                                 command=self.stop_video)
        self.update_video()  # Начинаем обновление видео

    def update_video(self):
        if self.cap is not None:
            ret, frame = self.cap.read()  # Читаем кадр из вебкамеры
            if ret:
                # Преобразуем BGR в RGB
                frame = cv2.resize(frame[0:480, 80:560], (256, 256))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Конвертируем в PIL формат
                self.img = Image.fromarray(frame)

                main_image.img_recognition(self)

            # Повторяем вызов этой функции через 1000 мс (1 секунда)
            self.root.after(10, self.update_video)

    def stop_video(self):
        self.cap.release()  # Освобождаем объект захвата видео
        self.cap = None
        self.video_button.config(text="Распознавать эмоции с видеопотока",
                                 command=self.start_video)
        self.img = Image.open("images/white.jpg")
        main_image.plot_image(self, "white")
        self.select_button.config(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.mainloop()
