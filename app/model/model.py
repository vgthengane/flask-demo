import tensorflow as tf 


class Classifier(tf.keras.Model):

    def __init__(self, num_outs=1):

        super(Classifier, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(16, (7, 7), activation="relu", padding="same")
        self.pool1 = tf.keras.layers.MaxPool2D((4, 4))

        self.conv2 = tf.keras.layers.Conv2D(32, (5, 5), activation="relu", padding="same")
        self.pool2 = tf.keras.layers.MaxPool2D((3, 3))

        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")
        self.pool3 = tf.keras.layers.MaxPool2D((2, 2))

        self.flat = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(num_outs, activation="sigmoid")

    
    def call(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))

        out = self.fc(self.flat(x))

        return out


# model = Classifier(num_outs=1)
# model(tf.keras.layers.Input(shape=(512, 1024, 3)))
# model.summary()



