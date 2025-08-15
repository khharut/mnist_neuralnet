from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, Flatten

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
<<<<<<< HEAD
model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
=======
model.add(Conv2D(64, kernel_size=3, activation='relu',
                 input_shape=(28, 28, 1)))

>>>>>>> 8980b8785f2133c75163de1d474803049b9e9990
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=3, verbose=2)
