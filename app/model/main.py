import os
import numpy as np 
# import argparse
from model import Classifier
from prepare_data import get_loaders

import tensorflow as tf


data_dir = "../data"
batch_size = 32
image_size = (512, 1024, 3)
phase = "test" 
image_URL = "../data/valid/handwritten/00002.png"

class_dict = {0: "handwritten",
              1: "printed"}

model = Classifier()

if phase == "train":

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    model.build((None,) + image_size)
    model.summary()


    train_loader, val_loader = get_loaders(data_dir, "train", image_size[:2], batch_size)

    model.fit(
        train_loader,
        steps_per_epoch=train_loader.samples // batch_size,
        epochs=5,
        validation_data=val_loader,
        validation_steps=val_loader.samples // batch_size
    )

    model.save_weights("../printed_and_handwritten_classifier.h5")

else:
    pass
    # model.build((None,) + image_size)
    # model.load_weights("../printed_and_handwritten_classifier.h5")
    # image = prepare_image(image_URL, image_size)
    # pred = model.predict(image)

    # key = 0 if pred[0][0] < 0.5 else 1
    # print(f"Predicted as \"{class_dict[key]}\" image.") 



    








