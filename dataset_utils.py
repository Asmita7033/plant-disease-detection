# dataset_utils.py
# Helpers to build tf.data datasets from folders, with augmentations.

import tensorflow as tf
import os

AUTOTUNE = tf.data.AUTOTUNE

def get_label_from_path(file_path, class_names):
    parts = tf.strings.split(file_path, os.path.sep)
    # expects .../train/class_name/image.jpg
    return tf.cast(tf.equal(class_names, parts[-2]), tf.float32)

def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32) # 0..1
    return img

def process_path(file_path, img_size, class_names):
    label = tf.strings.split(file_path, os.path.sep)[-2]
    label = tf.argmax(tf.cast(tf.equal(class_names, label), tf.int32))
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    img = tf.image.resize(img, img_size)
    return img, label

def build_dataset_from_folder(folder, img_size=(224,224), batch_size=32, shuffle=True):
    # find class names
    train_dir = folder
    class_names = sorted(next(os.walk(train_dir))[1])
    list_ds = tf.data.Dataset.list_files(str(train_dir + '/*/*'), shuffle=shuffle)
    process = lambda fp: process_path(fp, img_size, class_names)
    ds = list_ds.map(process, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds, class_names
