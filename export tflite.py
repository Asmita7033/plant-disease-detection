# export_tflite.py
# Convert saved Keras model to TFLite for Raspberry Pi/Edge usage.

import tensorflow as tf
import argparse
import json
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--h5', default='models/final_model.h5')
    p.add_argument('--out', default='models/model.tflite')
    p.add_argument('--quantize', action='store_true', help='Use post-training quantization (dynamic)')
    return p.parse_args()

def main():
    args = parse_args()
    model = tf.keras.models.load_model(args.h5)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if args.quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # For full integer quantization you'd need a representative dataset function
    tflite_model = converter.convert()
    with open(args.out, 'wb') as f:
        f.write(tflite_model)
    print("Saved TFLite to", args.out)

if __name__ == '__main__':
    main()
