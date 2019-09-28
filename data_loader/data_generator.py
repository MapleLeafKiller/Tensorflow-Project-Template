import numpy as np
from tensorflow.python.keras.datasets import mnist, cifar10
from tensorflow.python.keras.utils import to_categorical

class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        self.X_train = np.ones((500, 784))
        self.y_train = np.ones((500, 10))

    def load_mnist(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        print("shape: X_train:{},y_train:{},X_test:{},y_test:{}".
              format(X_train.shape, y_train.shape, X_test.shape, y_test.shape, ))

        # for tf.layers.conv2d, input has to be 4 dim:(batch,height,width,channels), for mnist channels=1
        X_train, X_test = np.expand_dims(X_train / 255.0, axis=-1), np.expand_dims(X_test / 255.0, axis=-1)
        y_train, y_test = to_categorical(y_train), to_categorical(y_test)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.X_train[idx], self.y_train[idx]
