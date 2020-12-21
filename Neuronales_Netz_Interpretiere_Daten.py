from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications.nasnet import preprocess_input
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import IO_Basic as io
from segmentation import segment_image

WEIGHT_PATH = './output/weights_felix.h5'
IMAGE_PATH = './ressource_slic_korrekte/Validation'
FOLDER_LENGTH = 10
IMAGE_SIZE = (224, 224)


def build_network(folder_number=FOLDER_LENGTH, image_size=IMAGE_SIZE):
    if folder_number is False:
        raise ValueError("Fehler ein Netzwerk mit Ausgangslayer 0 kann nicht erzeugt werden!\n")

    # Define Hyper parameters
    init_lr = 1e-4
    epochs = 10

    # Build Network
    pretrained_model = NASNetMobile(input_tensor=Input(shape=image_size + (3,)), weights='imagenet', include_top=False)

    # don't train existing weights
    for layer in pretrained_model.layers:
        layer.trainable = False
    # our added layers - you can add more if you want
    layer1 = pretrained_model.output
    layer1 = AveragePooling2D(pool_size=(2, 2))(layer1)
    layer1 = Flatten(name="flatten")(layer1)
    layer1 = Dense(128, activation="relu")(layer1)
    layer1 = Dropout(0.5)(layer1)
    layer1 = Dense(64, activation="relu")(layer1)
    layer1 = Dropout(0.5)(layer1)
    prediction = Dense(folder_number, activation="softmax")(layer1)

    # create a model object
    model = Model(inputs=pretrained_model.input, outputs=prediction)
    model.summary()

    # tell the model what cost and optimization method to use
    opt = Adam(lr=init_lr, decay=init_lr / epochs)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # Gewichte Laden
    model.load_weights(WEIGHT_PATH)

    return model


def load_model_weights_and_build_network(Model_Pfad='./output/Model_Own Dataset.h5',
                                         Weights_Pfad='./output/weights_felix.h5'):
    model = tf.keras.models.load_model(Model_Pfad)
    model.summary()
    model.load_weights(Weights_Pfad)
    return model


def test(image_path=IMAGE_PATH, image_size=IMAGE_SIZE):
    # useful for getting number of files
    test_image_files = glob(image_path + '/*/*.jp*g')
    print(f'Found {len(test_image_files)} images.')

    # useful for getting number of classes
    folders = glob(image_path + '/*')

    # Aufruf des Neuronalennetzwerk
    neuronal_network = build_network(len(folders))

    # image preprocessing and data augmentation during training
    datagen_test = ImageDataGenerator(fill_mode="nearest", preprocessing_function=preprocess_input)

    # Call Generators
    batch_size = 64
    test_generator = datagen_test.flow_from_directory(
        image_path,
        target_size=image_size,
        shuffle=True,
        batch_size=batch_size,
    )

    # Print all Labels
    labels = [None] * len(test_generator.class_indices)
    for k, v in test_generator.class_indices.items():
        labels[v] = k
    print(labels)

    for test_image in test_generator:
        predict = neuronal_network.predict(test_image)
        for i in range(len(test_image[0])):
            plt.imshow(test_image[0][i])
            plt.title("Solllabel: " + str(np.argmax(np.round(test_image[1][i], 2))) +
                      " Istlabel: " + str(np.argmax(predict[i])) +
                      "\nPrediction: " + str(np.round(predict[i], 2)))
            plt.show()
            # plt.waitforbuttonpress()


if __name__ == '__main__':
    model = load_model_weights_and_build_network()
    webcam = io.capture_webcam_open(0)
    # model = build_network()

    no_exit = True
    while no_exit:
        img0, img_gray = io.capture_webcam_multi_frame(webcam)
        # cv2.imshow('Bild', img0)
        # cv2.waitKey()
        img1 = segment_image(img0)
        # cv2.imshow('Segmentiertes Bild', img1)
        # cv2.waitKey()

        image_to_detect = (cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

        image_to_detect = tf.keras.applications.nasnet.preprocess_input(image_to_detect)

        image_to_detect = np.expand_dims(image_to_detect, axis=0)
        detection = model.predict(image_to_detect)
        value = np.amax(detection)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if value >= 0.65:
            cv2.putText(img1, str(np.argmax(detection)), (0, 200), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(img1, "ND", (0, 200), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        img2 = np.hstack([img0, img1])
        cv2.namedWindow("Prediction")
        # cv2.moveWindow("Prediction", 0,0)
        cv2.imshow('Prediction', img2)
        key = cv2.waitKey()
        print(str(detection))
        # print(key)
        if key == 113:  # Abfrage Taste q
            print("We will Leave the Loop!")
            no_exit = False

        cv2.destroyAllWindows()
    io.capture_webcam_close(webcam)
