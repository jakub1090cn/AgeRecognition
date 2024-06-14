import os
import pandas as pd
from sklearn.model_selection import train_test_split


test_dir = 'data/20-50/train'
paths = []
for image_class in os.listdir(test_dir):
    class_dir = os.path.join(test_dir, image_class, )
    if os.path.isdir(class_dir):
        for image in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image)
            print(image_path)
            paths.append(image_path)
train_data, val_data = train_test_split(paths, test_size=0.2)

train_df = pd.DataFrame(train_data, columns=['image_path'])
val_df = pd.DataFrame(val_data, columns=['image_path'])

train_df.to_csv('train_data.csv', index=False)
val_df.to_csv('val_data.csv', index=False)