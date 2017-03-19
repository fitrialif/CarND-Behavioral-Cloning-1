import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, ELU
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.advanced_activations import PReLU
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K

DATA_PATH = './data/'


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
                    #image = cv2.resize(cv2.imread(name), (200, 100), interpolation=cv2.INTER_AREA)
                    image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2YUV)
                    images.append(image)
                    #images.append(cv2.flip(image, 1))
                    #angles.append(angle * -1.0)
                correction = 0.2
                angle = float(batch_sample[3])
                angles.append(angle)
                angles.append(angle + correction)
                angles.append(angle - correction)

            yield shuffle(np.array(images), np.array(angles))


def compare_images(left_image, right_image):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(left_image)
    ax1.set_title('Shape ' + str(left_image.shape), fontsize=50)
    ax2.imshow(right_image)
    ax2.set_title('Shape ' + str(right_image.shape), fontsize=50)
    plt.show()

lines = []
with open('./data/driving_log.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)

# compile and train the model using the generator function
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# 1. define basic model
model = Sequential()
# 6. add cropping
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

model.summary()
#checkpoint = ModelCheckpoint("model-{epoch:02d}.h5", monitor='loss', verbose=1, save_best_only=False, mode='max')
#callbacks_list = [checkpoint]


model.compile(optimizer='adam', loss='mse')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*3, validation_data=validation_generator, nb_val_samples=len(validation_samples)*3, nb_epoch=3)
model.save('./model.h5')

#image = cv2.cvtColor(cv2.imread('./data_5/IMG/' + lines[0][2].split('\\')[-1]), cv2.COLOR_BGR2RGB)
#image = cv2.resize(image,(200, 100), interpolation=cv2.INTER_AREA)
#image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
#cropping_output = K.function([model.layers[0].input], [model.layers[0].output])

#cropped_image = cropping_output([image[None,...]])[0]
#compare_images(image, np.uint8(cropped_image.reshape(cropped_image.shape[1:])))
#compare_images(image, image2)
