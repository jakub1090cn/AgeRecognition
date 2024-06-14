import pandas as pd
import tensorflow as tf
import cv2

def my_data_generator_test():
    for _, row in test_data.iterrows():
        path = row['image_path']
        image = cv2.imread(path)
        image_resized = cv2.resize(image, IMG_RESOLUTION)
        image_normalized = image_resized / 255.0
        image_flattened = image_normalized.flatten()
        sample = image_flattened
        parts = path.split('\\')
        label = int(parts[-2]) - MIN_AGE
        yield sample, (label,)

test_data = pd.read_csv('test_data.csv')

IMG_RESOLUTION = (224, 224)
MIN_AGE = 20

batch_size = 32
x_shape = (IMG_RESOLUTION[0] * IMG_RESOLUTION[1] * 3,)
x_type = tf.float32
y_shape = (1,)
y_type = tf.int32

test_ds = tf.data.Dataset.from_generator(my_data_generator_test, output_signature=(
         tf.TensorSpec(shape=x_shape, dtype=x_type),
         tf.TensorSpec(shape=y_shape, dtype=y_type)))

test_ds = test_ds.batch(batch_size)

model = tf.keras.models.load_model('model.keras')

model.evaluate(test_ds)

