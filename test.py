import tensorflow as tf
import os

def is_valid_image(file_path):
    try:
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        return True
    except:
        print("Corrupt or invalid image:", file_path)
        return False

def clean_dataset(directory):
    for root, dirs, files in os.walk(directory):
        for f in files:
            path = os.path.join(root, f)
            if not is_valid_image(path):
                os.remove(path)
clean_dataset(output)