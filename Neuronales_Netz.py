# import the necessary packages
import os
import zipfile
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.layers import Dense, Flatten, Dropout, AveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

r = []
#################################################################
tf.test.is_gpu_available()
# Load Data
# Check if folder exists
# create data folder if not existing and extract data into it.
if not os.path.exists("./ressource"):
    os.makedirs("./ressource")

if not os.path.exists("./ressource/"):
    zip_ref = zipfile.ZipFile("Gesture_img.zip", 'r')
    zip_ref.extractall("./data/")
    zip_ref.close()

train_path = './ressource/Training'
valid_path = './ressource/Validation'
test_path = './ressource/Test'

# useful for getting number of files
image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(valid_path + '/*/*.jp*g')
test_image_files = glob(test_path + '/*/*.jp*g')

# useful for getting number of classes
folders = glob(train_path + '/*')

# look at an image for fun
plt.imshow(image.load_img(np.random.choice(image_files)))
# plt.show()

#########################################################
# Define Hyper parameters
INIT_LR = 1e-4
epochs = 1
batch_size = 64
# Define pre-build NASNet_Mobile Network or Mobilenet_V2
IMAGE_Size = (224, 224)

##########################################################
# Image generator
# image preprocessing and data augmentation during training

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False,
    shear_range=0.15,
    zoom_range=0.25,
    fill_mode="nearest")
##########################################


Pretrained_Model = NASNetMobile(input_tensor=Input(shape=IMAGE_Size + (3, )), weights='imagenet', include_top=False)
##########################################################

# don't train existing weights
for layer in Pretrained_Model.layers:
    layer.trainable = False
# our added layers - you can add more if you want
layer1 = Pretrained_Model.output
layer1 = AveragePooling2D(pool_size=(7, 7))(layer1)
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
model.compile(
    loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']
)

#########################
# Call Generators
train_generator = datagen.flow_from_directory(
    train_path,
    target_size=IMAGE_Size,
    shuffle=True,
    batch_size=batch_size,
)

valid_generator = datagen.flow_from_directory(
    valid_path,
    target_size=IMAGE_Size,
    shuffle=True,
    batch_size=batch_size,
)

test_generator = datagen.flow_from_directory(
    valid_path,
    target_size=IMAGE_Size,
    shuffle=True,
    batch_size=batch_size,
)
# Print all Labels
labels = [None] * len(train_generator.class_indices)
for k, v in train_generator.class_indices.items():
    labels[v] = k

print(labels)

# Start Training
NASNetMobile_callback = tf.keras.callbacks.ModelCheckpoint(filepath ="Mobilenet_Model_Checkpoint{epoch:04d}.ckpt",save_weights_only=True, verbose=1)
# fit the model
r = model.fit_generator(
    train_generator,
    validation_data=valid_generator,
    epochs=epochs,
    steps_per_epoch=len(image_files) // batch_size,
    validation_steps=len(valid_image_files) // batch_size,
    verbose=1,
    callbacks=NASNetMobile_callback
)

# saving the NASNet model
# saving the model
# save_dir = "/results/"
model_name = 'Model.h5'
weights_name = 'weights.h5'
model.save(model_name)
model.save_weights(weights_name)
print('Saved trained model at %s ' % model_name)
print('Saved weights at %s ' % weights_name)

# Show Accuracity and Loss
# plot some data

# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['accuracy'], label='train accuracy')
plt.plot(r.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()

################################
# # Perform Predictions on 5 Test Images
# for i in range(0, 5):
#     image_to_detect = (cv2.cvtColor((testX[i]), cv2.COLOR_BGR2RGB))
#     # plt.imshow(cv2.cvtColor((PictureX0[i]), cv2.COLOR_BGR2RGB))
#     # plt.show()
#     image_to_detect = cv2.resize(image_to_detect, (224, 224))
#     image_to_detect = preprocess_input(image_to_detect)
#     image_to_detect = np.expand_dims(image_to_detect, axis=0)
#     (mask, withoutMask) = model.predict(image_to_detect)[0]
#
#     label = "Mask Correct" if mask > withoutMask else "Mask NOT Correct"
#     # Prediction of the Image
#     label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
#     plt.imshow(cv2.cvtColor((testX[i]+1)/2, cv2.COLOR_BGR2RGB))
#     plt.title("Output: " + label)
#     plt.show()
# #######################################################
# # Prediction on unseen Data
# i = 0
# for i in range(0, 10):
#     img = cv2.imread(os.path.abspath('Unseen_Data/' + str(i) + '.jpg'))
#     image_to_detect = cv2.resize(img, (224, 224))
#     image_to_detect = preprocess_input(image_to_detect)
#     image_to_detect = np.expand_dims(image_to_detect, axis=0)
#     (mask, withoutMask) = model.predict(image_to_detect)[0]
#
#     label = "Mask Correct" if mask > withoutMask else "Mask NOT Correct"
#     # Prediction of the Image
#     label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.title("Output: " + label)
#     plt.show()
#
# #######################################################
# # Apply Predictions on Web cam Data
# def Webcam():
#     webcam = cv2.VideoCapture(0)
#     if webcam.isOpened():
#         true_, img = webcam.read()
#         webcam.release()
#         return cv2.resize(img, (224, 224))
#     else:
#         return cv2.resize(np.zeros((200, 200, 3), np.uint8), (224, 224))
#
# img1 = Webcam()
# # img = np.expand_dims(np.stack(img1, axis=0), axis=0)
# image_to_detect = (cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
# image_to_detect = img_to_array(image_to_detect)
# image_to_detect = preprocess_input(image_to_detect)
# image_to_detect = np.expand_dims(image_to_detect, axis=0)
#
# # Prediction of the Image
# (mask, withoutMask) = model.predict(image_to_detect)[0]
# label = "Mask Correct" if mask > withoutMask else "Mask NOT Correct"
# label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
# plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
# plt.title("Output: " + label)
# plt.show()
