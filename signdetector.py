import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils as np_utils
from tensorflow.keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import matplotlib.pyplot as plt
import sys


class SignDetector:
    def __init__(self, X, y, n_classes):
        self.X = X
        self.y = y
        self.n_classes = n_classes
        self.model = None

    def train(self, epochs=30):
        try:
            model = load_model("./models/model")
            return model
        except FileNotFoundError:
            pass
        except IOError:
            pass

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
        X_train = X_train.astype('float32') / 255
        X_test = X_test.astype('float32') / 255

        y_train = np_utils.to_categorical(y_train, self.n_classes)
        y_test = np_utils.to_categorical(y_test, self.n_classes)
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
        model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dropout(0.5))

        model.add(Dense(self.n_classes, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs)

        self.summarize_diagnostics(history)

        model.save('./models/model')

        return model

    def summarize_diagnostics(self, history):
        plt.subplot(121)
        plt.plot(history.history['loss'], color='blue', label='train')
        plt.plot(history.history['val_loss'], color='orange', label='test')
        plt.title('Loss function')
        plt.xlabel('Epoch')
        plt.legend()

        plt.subplot(122)
        plt.plot(history.history['accuracy'], color='blue', label='train')
        plt.plot(history.history['val_accuracy'], color='orange', label='test')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()

        filename = sys.argv[0].split('/')[-1]
        plt.savefig(filename + '_plot.png')
        plt.show()
        plt.close()

def create_model(X, y, n_classes):
    try:
        model = load_model("./models/model")
    except FileNotFoundError:
        model = SignDetector(X, y, n_classes)
    except IOError:
        model = SignDetector(X, y, n_classes)

    return model