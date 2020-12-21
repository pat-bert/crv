import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.nasnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from Neuronales_Netz_Interpretiere_Daten import load_model_weights_and_build_network

if __name__ == '__main__':
    model = load_model_weights_and_build_network()

    # Start
    IMAGE_SIZE = (224, 224)
    test_path = './Felix_ressource_segmented/Test'
    batch_size = 64

    # Image Generator
    test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_data = test_gen.flow_from_directory(test_path, target_size=IMAGE_SIZE, batch_size=batch_size, shuffle=False)

    # Confution Matrix and Classification Report
    Y_pred = model.predict(test_data)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(test_data.classes, y_pred))
    print('Classification Report')
    print(classification_report(test_data.classes, y_pred))
