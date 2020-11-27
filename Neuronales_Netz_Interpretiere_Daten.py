# import the necessary packages
import os
import zipfile
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import IO_Basic
import cv2

def build_network(folder_number = 0):

    if folder_number is False:
        print("Fehler ein Netzwerk mit Ausgangslayer 0 kann nicht erzeugt werden!\n")
        return None

    # Define Hyper parameters
    INIT_LR = 1e-4
    epochs = 10
    batch_size = 64
    # Define pre-build NASNet_Mobile Network or Mobilenet_V2
    IMAGE_Size = (224, 224)

    #Build Network
    Pretrained_Model = NASNetMobile(input_tensor=Input(shape=IMAGE_Size + (3,)), weights='imagenet', include_top=False)


    # don't train existing weights
    for layer in Pretrained_Model.layers:
        layer.trainable = False
    # our added layers - you can add more if you want
    layer1 = Pretrained_Model.output
    layer1 = MaxPooling2D(pool_size=(7, 7))(layer1)
    layer1 = Flatten(name="flatten")(layer1)
    layer1 = Dense(128, activation="relu")(layer1)
    layer1 = Dropout(0.5)(layer1)
    layer1 = Dense(64, activation="relu")(layer1)
    layer1 = Dropout(0.5)(layer1)
    prediction = Dense(len(folders), activation="softmax")(layer1)

    # create a model object
    model = Model(inputs=Pretrained_Model.input, outputs=prediction)

    #   structure of the Model
    model.summary()
    # tell the model what cost and optimization method to use
    opt = Adam(lr=INIT_LR, decay=INIT_LR / epochs)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    #Gewichte Laden
    model.load_weights('./output/weights.h5')

    return model



#################################################################
# Load Data
# Check if folder exists
# create data folder if not existing and extract data into it.
if not os.path.exists("./ressource"):
    os.makedirs("./ressource")

# if not os.path.exists("./ressource/"):
#     zip_ref = zipfile.ZipFile("Gesture_img.zip", 'r')
#     zip_ref.extractall("./data/")
#     zip_ref.close()

test_path = './ressource/Test'

# useful for getting number of files
test_image_files = glob(test_path + '/*/*.jp*g')

# useful for getting number of classes
folders = glob(test_path + '/*')

#Aufruf des Neuronalennetzwerk
Neuronalesnetzwerk = build_network(len(folders))

#Define Batch_Size and Image Size
batch_size = 64
IMAGE_Size = (224, 224)

##########################################################
# Image generator
# image preprocessing and data augmentation during training
datagen_test = ImageDataGenerator(
    rescale=1. / 255,
    fill_mode="nearest")

#########################
# Call Generators
test_generator = datagen_test.flow_from_directory(
    test_path,
    target_size=IMAGE_Size,
    shuffle=True,
    batch_size=batch_size,
)

# Print all Labels
labels = [None] * len(test_generator.class_indices)
for k, v in test_generator.class_indices.items():
    labels[v] = k

print(labels)

for test in test_generator:
    predict = Neuronalesnetzwerk.predict(test)
    #print(test[0][0])
    #print(predict[0])
    for i in range(len(test[0])):
        plt.imshow(test[0][i])
        plt.title("Solllabel: " + str(np.round(test[1][i], 2)) + "\nPrediction: " + str(np.round(predict[i], 2)))
        plt.show()
        plt.waitforbuttonpress()

# rgb_frame, gray_frame = IO_Basic.capture_webcam()
# rgb_resize = cv2.resize(rgb_frame, IMAGE_Size)
# test = tf.keras.applications.nasnet.preprocess_input(np.expand_dims(rgb_resize, axis=0))
# predict = Neuronalesnetzwerk.predict(test)
# plt.imshow(test)
# plt.title("Prediction: " + str(np.round(predict, 2)))
# plt.show()
# plt.waitforbuttonpress()