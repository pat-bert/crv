# import the necessary packages
import os
import zipfile
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications.nasnet import preprocess_input
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, AveragePooling2D
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

train_path = './Felix_ressource_segmented/Training'
valid_path = './Felix_ressource_segmented/Validation'
test_path = './Felix_ressource_segmented/Test'

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
epochs = 50
batch_size = 64
# Define pre-build NASNet_Mobile Network or Mobilenet_V2
IMAGE_Size = (224, 224)

##########################################################
# Image generator
# image preprocessing and data augmentation during training

datagen_train = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False,
    shear_range=0.15,
    zoom_range=0.25,
    # rescale=1. / 255,
    fill_mode="nearest",
    preprocessing_function=preprocess_input)

datagen_valid = ImageDataGenerator(
    # rescale=1. / 255,
    fill_mode="nearest",
    preprocessing_function=preprocess_input)

datagen_test = ImageDataGenerator(
    # rescale=1. / 255,
    fill_mode="nearest",
    preprocessing_function=preprocess_input)
##########################################

title = 'Own Dataset'

Pretrained_Model = NASNetMobile(input_tensor=Input(shape=IMAGE_Size + (3,)), weights='imagenet', include_top=False)
##########################################################
# don't train existing weights
Pretrained_Model.trainable = False

# our added layers - you can add more if you want
layer1 = Pretrained_Model.output
# Pooling layer
layer1 = AveragePooling2D(pool_size=2)(layer1)
layer1 = Flatten(name="flatten")(layer1)
# Dense layers with dropout
layer1 = Dense(128, activation="relu")(layer1)
layer1 = Dropout(0.5)(layer1)
layer1 = Dense(64, activation="relu")(layer1)
layer1 = Dropout(0.5)(layer1)
prediction = Dense(len(folders), activation="softmax")(layer1)

# create a model object
model = Model(inputs=Pretrained_Model.input, outputs=prediction)
model.summary()

# tell the model what cost and optimization method to use
opt = Adam(lr=INIT_LR, decay=INIT_LR / epochs)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# model.load_weights('./output/weights.h5')

#########################
# Call Generators
train_generator = datagen_train.flow_from_directory(
    train_path,
    target_size=IMAGE_Size,
    shuffle=True,
    batch_size=batch_size,
)

validation_generator = datagen_valid.flow_from_directory(
    valid_path,
    target_size=IMAGE_Size,
    shuffle=True,
    batch_size=batch_size,
)

test_generator = datagen_test.flow_from_directory(
    test_path,
    target_size=IMAGE_Size,
    shuffle=False,
    batch_size=batch_size,
)
# Print all Labels
labels = [None] * len(train_generator.class_indices)
for k, v in train_generator.class_indices.items():
    labels[v] = k

print(labels)

# Start Training
NASNetMobile_callback = tf.keras.callbacks.ModelCheckpoint(filepath="Mobilenet_Model_Checkpoint{epoch:04d}.ckpt",
                                                           save_weights_only=True, verbose=1)

r = model.fit(
    x=train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    batch_size=batch_size,
    steps_per_epoch=len(image_files) // batch_size,
    validation_steps=len(valid_image_files) // batch_size,
    verbose=1,
    callbacks=[NASNetMobile_callback],
    use_multiprocessing=False
)

# saving the NASNet model
# saving the model
# save_dir = "/results/"
model_name = f'Model_{title}.h5'
weights_name = f'weights_{title}.h5'
model.save(model_name)
model.save_weights(weights_name)
print('Saved trained model at %s ' % model_name)
print('Saved weights at %s ' % weights_name)

# Show Accuracity and Loss
# plot some data

# loss
plt.figure()
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.ylim(bottom=0.0)
plt.grid()
plt.title(title)
plt.show()

plt.figure()
plt.plot(r.history['accuracy'], label='train accuracy')
plt.plot(r.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.ylim((0.0, 1.0))
plt.grid()
plt.title(title)
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
