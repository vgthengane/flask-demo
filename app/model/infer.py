import tensorflow as tf
from app.model.model import Classifier
from app.model.prepare_data import prepare_image


def predict(image):

    image_size = (512, 1024, 3)
    class_dict = {
        0: "handwritten",
        1: "printed"
    }

    model = Classifier()
    model.build((None,) + image_size)
    model.load_weights("../printed_and_handwritten_classifier.h5")
    image = prepare_image(image, image_size)
    pred = model.predict(image)

    preds = 0 if pred[0][0] < 0.5 else 1
    return preds, class_dict[preds]



