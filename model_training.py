import os
#hides tensorflow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist

print("Imports done")

#(teaching images, corresponding numbers), (testing images, corresponding numbers)    
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Data loaded")

x_train = x_train.reshape(-1, 28,28, 1) / 255.0
x_test = x_test.reshape(-1, 28,28, 1) / 255.0

print("Data preprocessed")

#model = variable holding neural network
model = Sequential([
    #simple patterns
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(),
    #more complex patterns
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    #fully connects neurons, 128 = number of neurons
    Dense(128, activation='relu'),
    #10 neurons (0-9), softmax = outputs probabilities for each neuron 
    Dense(10, activation='softmax')
])

print("Model created")

'''
'adam'= algorithm for adjusting weights
'sparse_categorical_crossentropy' = How the model measures how wrong it is
'''
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("model compiled, started training...")

'''
model.fit() = start training
epochs = number of times the model will see the entire training dataset
validation_data = after each epoch test to see how good the model is
'''
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

print("training done, saving...")

#saves the model to a file
model.save('digit_model.h5')

print("model saved")