import os
import cv2
import glob
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from train_classification import train_custom


def load_data(data_path, data_folders, face_images, face_labels):
    """

    :param data_path:
    :param data_folders:
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


if __name__ == "__main__":
    data_path = "/path/to/preprocessed_dataset"
    data_folders = ["mask", "wrong_mask", "wrong_mask_aug", "no_mask", "no_mask_aug"]

    face_images = []
    face_labels = []

    dataset_images, dataset_labels = load_data(data_path, data_folders, face_images, face_labels)
    train_custom(dataset_images, dataset_labels)
