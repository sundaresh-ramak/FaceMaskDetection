import numpy as np
import cv2
import glob
import os
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import yolo_face_detect


mask_detect_model = load_model('/path/to/trained_model.h5')

face_model = load_model('/path/to/pretrained/faceDetection/yolo_face_model.h5')

images_path = '/path/to/faceMaskClassifcation/testDataset'

result_dir = '/path/to/prediction_results'


def face_mask_predict(file_path):
    """

    :param file_path: image file path for face and mask detection
    :return:
    """
    image = cv2.imread(file_path)

    _, filename = os.path.split(file_path)
    destination_path = os.path.join(result_dir, filename)

    face_detections_boxes = yolo_face_detect.get_faces(file_path, face_model)

    for i in range(len(face_detections_boxes)):
        box = face_detections_boxes[i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        start_point = (x1, y1)
        end_point = (x2, y2)
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.45
        face = image[y1:y2, x1:x2]
        if face.size == 0:
            continue

        if face.shape[0] > 224 or face.shape[1] > 224:
            face = cv2.resize(face, (224, 224))

        resized_image = np.zeros((224, 224, 3), np.uint8)
        h, w = face.shape[:2]
        yoff = round((224 - h) / 2)
        xoff = round((224 - w) / 2)

        resized_image[yoff:yoff + h, xoff:xoff + w] = face

        face = resized_image.astype(np.float32)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        (improper, mask, withoutMask) = mask_detect_model.predict(face)[0]

        label_id = np.argmax([improper, mask, withoutMask])

        if label_id == 1:
            label = "Mask"
            color = (255, 0, 0)
            max_val = mask
        elif label_id == 2:
            label = "No mask"
            color = (0, 255, 0)
            max_val = withoutMask
        else:
            label = "Improper"
            color = (0, 0, 255)
            max_val = improper

        label = "{}: {:.2f}%".format(label, max_val * 100)

        cv2.rectangle(image, start_point, end_point, color, thickness)
        cv2.putText(image, label, (x1, y1), font,
                    fontScale, color, thickness, cv2.LINE_AA)
    print("Prediction result saved at ", destination_path)
    cv2.imwrite(destination_path, image)


if __name__ == "__main__":
    isFile = os.path.isfile(images_path)
    print("Starting face mask classification...")
    if isFile:
        face_mask_predict(images_path)
    else:
        for extension in ['*.png', '*.jpg', '*.jpeg', '*.tif']:
            for file_path in tqdm(glob.glob(os.path.join(images_path, extension))):
                face_mask_predict(file_path)

    print("Face mask classification complete...")
