import pandas as pd
import os
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping


def my_data_generator_train():
    for _, row in train_data.iterrows():
        path = row['image_path']
        image = cv2.imread(path)
        image_resized = cv2.resize(image, IMG_RESOLUTION)
        image_normalized = image_resized / 255.0
        image_flattened = image_normalized.flatten()
        sample = image_flattened
        parts = path.split('\\')
        label = int(parts[-2]) - MIN_AGE
        yield sample, (label,)


def my_data_generator_val():
    for _, row in val_data.iterrows():
        path = row['image_path']
        image = cv2.imread(path)
        image_resized = cv2.resize(image, IMG_RESOLUTION)
        image_normalized = image_resized / 255.0
        image_flattened = image_normalized.flatten()
        sample = image_flattened
        parts = path.split('\\')
        label = int(parts[-2]) - MIN_AGE
        yield sample, (label,)

train_data = pd.read_csv('train_data.csv')
val_data = pd.read_csv('val_data.csv')

IMG_RESOLUTION = (224, 224)
MIN_AGE = 20

batch_size = 32
x_shape = (IMG_RESOLUTION[0] * IMG_RESOLUTION[1] * 3,)
x_type = tf.float32
y_shape = (1,)
y_type = tf.int32

train_ds = tf.data.Dataset.from_generator(my_data_generator_train, output_signature=(
         tf.TensorSpec(shape=x_shape, dtype=x_type),
         tf.TensorSpec(shape=y_shape, dtype=y_type)))

train_ds = train_ds.shuffle(100)
train_ds = train_ds.batch(batch_size)

val_ds = tf.data.Dataset.from_generator(my_data_generator_val, output_signature=(
         tf.TensorSpec(shape=x_shape, dtype=x_type),
         tf.TensorSpec(shape=y_shape, dtype=y_type)))

val_ds = val_ds.batch(batch_size)

L2_REG = regularizers.l2(0.01)

model = models.Sequential([
    tf.keras.layers.Input(shape=x_shape),
    tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=L2_REG),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=L2_REG),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=L2_REG),
    tf.keras.layers.Dense(1, activation='relu')
])
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[early_stopping])

model.save('model.keras')