import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, ELU
from keras.layers.convolutional import Convolution2D, Cropping2D


# Use generator to avoid low memory
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = './data/IMG/' + batch_sample[i].split('\\')[-1]
                    image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2YUV)
                    images.append(image)
                correction = 0.2
                angle = float(batch_sample[3])
                # angle center image
                angles.append(angle)
                # angle left image
                angles.append(angle + correction)
                # angle right image
                angles.append(angle - correction)

            yield shuffle(np.array(images), np.array(angles))

# import driving log
lines = []
with open('./data/driving_log.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)

# compile and train the model using the generator function
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Define model
model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255 - 0.5, trainable=False))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid"))
model.add(ELU())
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid"))
model.add(ELU())
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid"))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, border_mode="valid"))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, border_mode="valid"))
model.add(Dropout(0.2))
model.add(ELU())
model.add(Flatten())
model.add(Dense(100))
model.add(ELU())
model.add(Dense(50))
model.add(ELU())
model.add(Dense(10))
model.add(ELU())
model.add(Dense(1, trainable=False))

# print model summary
model.summary()

# compile, train and save the model
model.compile(optimizer='adam', loss='mse')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*3, validation_data=validation_generator, nb_val_samples=len(validation_samples)*3, nb_epoch=3)
model.save('./model.h5')
