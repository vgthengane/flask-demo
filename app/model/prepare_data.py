import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_loaders(data_dir, phase, image_size, batch_size):

    train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True
            )

    test_datagen = ImageDataGenerator(
                    rescale=1./255
            )

    train_loader = train_datagen.flow_from_directory(
            os.path.join(data_dir, "train"),
            target_size=image_size,
            batch_size=batch_size,
            class_mode='binary')

    val_loader = test_datagen.flow_from_directory(
            os.path.join(data_dir, "valid"),
            target_size=image_size,
            batch_size=batch_size,
            class_mode='binary')


    return train_loader, val_loader


def prepare_image(image, image_size):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size[1], image_size[0]))
    image = np.array(image, dtype=np.float32)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    return image



