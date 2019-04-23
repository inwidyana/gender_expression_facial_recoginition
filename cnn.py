from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from PIL import Image, ImageFile
import os

currentDirectory = os.path.dirname(os.path.realpath(__file__))

ImageFile.LOAD_TRUNCATED_IMAGES = True

classifier = Sequential()

classifier.add(Conv2D(100, (1, 1), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (3, 3)))
classifier.add(Conv2D(100, (1, 1), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(3, 3)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 64, activation='relu'))
classifier.add(Dense(units = 4, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(currentDirectory + '/data/test', target_size = (64, 64), batch_size = 32, class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(currentDirectory + '/data/train', target_size = (64, 64), batch_size = 32, class_mode='categorical')

classifier.fit_generator(training_set, steps_per_epoch = 100, epochs = 4, validation_data = test_set, validation_steps = 10)