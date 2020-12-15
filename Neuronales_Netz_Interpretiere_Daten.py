from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications.nasnet import preprocess_input
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from segmentation import segment_image
import tensorflow as tf

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


#######################################################
# Apply Predictions on Web cam Data
def predict_webcam():
    webcam = cv2.VideoCapture(1)
    if webcam.isOpened():
        true_, img = webcam.read()
        webcam.release()
        return cv2.resize(img, (224, 224))
    else:
        return cv2.resize(np.zeros((200, 200, 3), np.uint8), (224, 224))


if __name__ == '__main__':

    model = build_network()
    while True:
        img0 = predict_webcam()
        img1 = segment_image(img0)
        #cv2.imshow('Segmentiertes Bild', img1)
        #cv2.waitKey()
        # img = np.expand_dims(np.stack(img1, axis=0), axis=0)
        image_to_detect = (cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

        image_to_detect = tf.keras.applications.nasnet.preprocess_input(image_to_detect)

        # image_to_detect = img_to_array(image_to_detect)
        # image_to_detect = preprocess_input(image_to_detect)
        image_to_detect = np.expand_dims(image_to_detect, axis=0)
        detection = model.predict(image_to_detect)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img1, str(np.argmax(detection)), (0, 200), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        img2 = np.hstack([img0, img1])
        cv2.imshow('Prediction', img2)
        print(str(detection))
        cv2.waitKey()
        cv2.destroyAllWindows()
        #Prediction of the Image



# # Alt vor dem Merge
# #################################################################
# # Load Data
# # Check if folder exists
# # create data folder if not existing and extract data into it.
# if not os.path.exists("./ressource"):
#     os.makedirs("./ressource")
#
# # if not os.path.exists("./ressource/"):
# #     zip_ref = zipfile.ZipFile("Gesture_img.zip", 'r')
# #     zip_ref.extractall("./data/")
# #     zip_ref.close()
#
# test_path = './ressource/Test'
#
# # useful for getting number of files
# test_image_files = glob(test_path + '/*/*.jp*g')
#
# # useful for getting number of classes
# folders = glob(test_path + '/*')
#
# #Aufruf des Neuronalennetzwerk
# Neuronalesnetzwerk = build_network(len(folders))
#
# #Define Batch_Size and Image Size
# batch_size = 64
# IMAGE_Size = (224, 224)
#
# ##########################################################
# # Image generator
# # image preprocessing and data augmentation during training
# datagen_test = ImageDataGenerator(
#     rescale=1. / 255,
#     fill_mode="nearest")
#
# #########################
# # Call Generators
# test_generator = datagen_test.flow_from_directory(
#     test_path,
#     target_size=IMAGE_Size,
#     shuffle=True,
#     batch_size=batch_size,
# )
#
# # Print all Labels
# labels = [None] * len(test_generator.class_indices)
# for k, v in test_generator.class_indices.items():
#     labels[v] = k
#
# print(labels)
#
# # for test in test_generator:
# #     predict = Neuronalesnetzwerk.predict(test)
# #     #print(test[0][0])
# #     #print(predict[0])
# #     for i in range(len(test[0])):
# #         plt.imshow(test[0][i])
# #         plt.title("Solllabel: " + str(np.round(test[1][i], 2)) + "\nPrediction: " + str(np.round(predict[i], 2)))
# #         plt.show()
# #         plt.waitforbuttonpress()
#
# rgb_frame, gray_frame = IO_Basic.capture_webcam()
# rgb_resize = cv2.resize(rgb_frame, IMAGE_Size)
# test = tf.keras.applications.nasnet.preprocess_input(rgb_resize)
# test = np.expand_dims(test, axis=0)
# predict = Neuronalesnetzwerk.predict(test)
# plt.imshow(test)
# plt.title("Prediction: " + str(np.round(predict, 2)))
# plt.show()
# plt.waitforbuttonpress()
