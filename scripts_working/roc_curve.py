import numpy as np
import os
import glob
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input


data_path = "/rootDir/to/faceMaskDataset/preprocessed"
data_folders = ["mask", "wrong_mask", "wrong_mask_aug", "no_mask", "no_mask_aug"]
classification_model = load_model("/path/to/trained_classification_model")


def load_data(data_path, data_folders, face_images, face_labels):
    """
    function to load the data from subfolders in preprocessed images
    :param data_path: rootDir of preprocessed dataset
    :param data_folders: subfolders containing classwise image crops
    :param face_images:
    :param face_labels:
    :return:
    """
    for data_folder in data_folders:
        sub_folder = os.path.join(data_path, data_folder)
        for image_name in glob.glob(os.path.join(sub_folder, '*.png')):
            image = cv2.imread(image_name)
            if image is None:
                continue

            if data_folder == 'mask':
                label = "with_mask"

            elif data_folder == 'wrong_mask':
                label = "mask_weared_incorrect"

            elif data_folder == 'wrong_mask_aug':
                label = "mask_weared_incorrect"

            else:
                label = "without_mask"

            image = image.astype(np.float32)
            image = preprocess_input(image)
            face_images.append(image)
            face_labels.append(label)

    face_images = np.array(face_images, dtype="float32")
    face_labels = np.array(face_labels)
    return face_images, face_labels


def calculate_metrics(input_images, input_labels, keras_model):
    lab_encoder = LabelEncoder()
    transformed_labels = lab_encoder.fit_transform(input_labels)
    transformed_labels = to_categorical(transformed_labels, num_classes=3)
    (train_imgs, test_imgs, train_labels, test_labels) = train_test_split(input_images, transformed_labels,
                                                                          test_size=0.1, stratify=transformed_labels,
                                                                          shuffle=25)

    print("Evaluating the trained classification model...")

    prediction_probs = keras_model.predict(test_imgs)
    predicted_classes = np.argmax(prediction_probs, axis=1)

    print(classification_report(test_labels.argmax(axis=1), predicted_classes))


if __name__ == "__main__":
    dataset_images, dataset_labels = load_data(data_path, data_folders, face_images=[], face_labels=[])
    calculate_metrics(dataset_images, dataset_labels, classification_model)
