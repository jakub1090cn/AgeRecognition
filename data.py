import os
import pandas as pd
from sklearn.model_selection import train_test_split
from glob import glob


train_dir = 'data/20-50/train'
test_dir = 'data/20-50/test'

paths = [image_path for class_dir in glob(os.path.join(train_dir, '*')) if os.path.isdir(class_dir) for image_path in glob(os.path.join(class_dir, '*'))]

train_data, val_data = train_test_split(paths, test_size=0.2)

test_data = [image_path for class_dir in glob(os.path.join(test_dir, '*')) if os.path.isdir(class_dir) for image_path in glob(os.path.join(class_dir, '*'))]

print(f"Number of training images: {len(train_data)}")
print(f"Number of validation images: {len(val_data)}")
print(f"Number of testing images: {len(test_data)}")

train_df = pd.DataFrame(train_data, columns=['image_path'])
val_df = pd.DataFrame(val_data, columns=['image_path'])
test_df = pd.DataFrame(test_data, columns=['image_path'])
train_df.to_csv('train_data.csv', index=False)
val_df.to_csv('val_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)