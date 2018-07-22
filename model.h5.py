import csv
import cv2
import numpy as np


def nvidiacnn(x_train, y_train):

    import keras
    from keras import regularizers
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda, Dropout
    from keras.layers.convolutional import Convolution2D, Cropping2D


    '''Create NVIDIA architecture model for cloning'''
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20),(0,0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2),activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2),activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2),activation='relu'))
    model.add(Convolution2D(64,3,3, subsample=(1,1),activation='relu'))
    model.add(Convolution2D(64,3,3, subsample=(1,1),activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(x_train,y_train,validation_split=0.2,shuffle=True, nb_epoch=5)
    model.save('model.h5')

    return

def readImageRGB(image_file_name):

    srcBGR = cv2.imread(image_file_name)

    if srcBGR is None:
        print('Unable to process the image: ' + image_file_name)
        exit(0)

    image = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)

    height, width, channels = image.shape
    if height == 0 or width == 0 or channels == 0:
        print(image_file_name)
        exit(0)

    return image

def preprocessTrain(lines):
    images = []
    measurements = []
    for line in lines:
        source_dir = './data/data/'
        center_image_file_name = source_dir + line[0]
        left_image_file_name = source_dir + line[1]
        right_image_file_name = source_dir + line[2]

        measurement = float(line[3])
        # print(measurement)

        left_measurement = measurement - 0.15
        # print(left_measurement)

        right_measurement = measurement + 0.15
        # print(right_measurement)

        # print(center_image_file_name)

        center_image = readImageRGB(center_image_file_name)
        left_image = readImageRGB(left_image_file_name)
        right_image = readImageRGB(right_image_file_name)

        images.append(center_image)
        images.append(left_image)
        images.append(right_image)

        measurements.append(measurement)
        measurements.append(right_measurement)
        measurements.append(left_measurement)

    print(len(images))
    print(len(measurements))

    x_train = np.array(images)
    y_train = np.array(measurements)

    nvidiacnn(x_train, y_train)

    return

lines = []
with open('./data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    #skip the header
    next(reader)
    for line in reader:
        lines.append(line)

    preprocessTrain(lines)

