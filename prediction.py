import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model('model.keras')

IMAGE_PATH = 'data/ncdc.jpg'
IMG_RESOLUTION = (224, 224)

image = cv2.imread(IMAGE_PATH)
image_resized = cv2.resize(image, IMG_RESOLUTION)
image_normalized = image_resized / 255.0

image_flattened = image_normalized.flatten()

image_batch = np.expand_dims(image_flattened, axis=0)

predictions = model.predict(image_batch)

print(predictions)
