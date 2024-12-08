from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras import models

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf

train_path = 'dataset/train'
val_path = 'dataset/test'

train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_gen = train_datagen.flow_from_directory(
    train_path,
    target_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    class_mode='sparse')

val_gen = val_datagen.flow_from_directory(
    val_path,
    target_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    class_mode='sparse')

emotions = list(train_gen.class_indices.keys())
print(emotions)

model = models.Sequential()

# block 1
model.add(layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', activation='elu',
                        input_shape=(48, 48, 1)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', activation='elu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2), padding='same'))
model.add(layers.Dropout(0.5))

# block 2
model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='elu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='elu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2), padding='same'))
model.add(layers.Dropout(0.5))

# block 3
model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', activation='elu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', activation='elu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2), padding='same'))
model.add(layers.Dropout(0.5))

# block 4
model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', activation='elu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', activation='elu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2), padding='same'))
model.add(layers.Dropout(0.5))

# block 5
model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', activation='elu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', activation='elu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2), padding='same'))
model.add(layers.Dropout(0.5))

# Block 6
model.add(layers.Flatten())
model.add(layers.Dense(256, kernel_initializer='he_normal', activation='elu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

# block 7
model.add(layers.Dense(128, kernel_initializer='he_normal', activation='elu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

# block 8
model.add(layers.Dense(64, kernel_initializer='he_normal', activation='elu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

# output layer with 7 classes
model.add(layers.Dense(7, activation='softmax'))


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


checkpoint = ModelCheckpoint('my_model.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=5,
                              verbose=1,
                              min_lr=1e-7)
callbacks = [checkpoint, reduce_lr]


history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=40,
                    verbose=1,
                    callbacks=callbacks)

# Сохранение всей модели
model.save('test_model1.keras')


import matplotlib.pyplot as plt

# График потерь
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# График точности
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
