# importing the required libraries
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

# loading data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
print(X_train.shape)
print(X_test.shape)
# reshaping data
#X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
#X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
# checking the shape after reshaping
print(X_train.shape)
print(X_test.shape)
# normalizing the pixel values
X_train = X_train / 255
X_test = X_test / 255

train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)
# defining model

from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

model = Sequential()
# adding convolution layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# adding pooling layer
model.add(MaxPool2D(2, 2))
# adding fully connected layer
model.add(Flatten())
model.add(Dense(100, activation='relu'))
# adding output layer
model.add(Dense(10, activation='softmax'))
model.summary()
# compiling the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fitting the model
model.fit(X_train, y_train, epochs=2)
#evaluting the model
model.evaluate(X_test,y_test)




