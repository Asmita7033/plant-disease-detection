# train.py
# Full training pipeline. Adjust paths and hyperparams as needed.

import os
import argparse
import tensorflow as tf
from tensorflow import keras
from dataset_utils import build_dataset_from_folder
from model import build_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', default='plantVillageData/train', type=str)
    p.add_argument('--val_dir', default='plantVillageData/val', type=str)
    p.add_argument('--img_size', default=224, type=int)
    p.add_argument('--batch_size', default=32, type=int)
    p.add_argument('--epochs', default=10, type=int)
    p.add_argument('--out_dir', default='models', type=str)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Building datasets...")
    train_ds, class_names = build_dataset_from_folder(args.train_dir, img_size=(args.img_size, args.img_size), batch_size=args.batch_size, shuffle=True)
    val_ds, _ = build_dataset_from_folder(args.val_dir, img_size=(args.img_size, args.img_size), batch_size=args.batch_size, shuffle=False)

    num_classes = len(class_names)
    print("Classes:", class_names)

    model = build_model(num_classes, img_size=(args.img_size, args.img_size, 3), base_trainable=False)

    # callbacks
    ckpt_path = os.path.join(args.out_dir, 'best_model.h5')
    checkpoint = keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_accuracy', mode='max')
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[checkpoint, reduce_lr, early]
    )

    print("Saving final model to", os.path.join(args.out_dir, 'final_model.h5'))
    model.save(os.path.join(args.out_dir, 'final_model.h5'))

    # Also save class names for inference
    import json
    with open(os.path.join(args.out_dir, 'class_names.json'), 'w') as f:
        json.dump(class_names, f)

    print("Training complete.")

if __name__ == '__main__':
    main()
