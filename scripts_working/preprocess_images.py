import cv2
import os
import glob
import xmltodict
import numpy as np

annotations_path = "D:/Projects/FaceMaskClassification/data/annotations"
images_path = "D:/Projects/FaceMaskClassification/data/images"

mask_images = "D:/Projects/FaceMaskClassification/data/mask"
incorrect_mask_images = "D:/Projects/FaceMaskClassification/data/wrong_mask"
no_mask_images = "D:/Projects/FaceMaskClassification/data/no_mask"

if not os.path.exists(mask_images):
    os.mkdir(mask_images)
if not os.path.exists(incorrect_mask_images):
    os.mkdir(incorrect_mask_images)
if not os.path.exists(no_mask_images):
    os.mkdir(no_mask_images)


def read_annotation_file(label_path):
    x = xmltodict.parse(open(label_path, 'rb'))
    item_list = x['annotation']['object']

    if not isinstance(item_list, list):
        item_list = [item_list]

    result = []

    for item in item_list:
        mask_status = item['name']
        bbox = [(int(item['bndbox']['xmin']), int(item['bndbox']['ymin'])),
                (int(item['bndbox']['xmax']), int(item['bndbox']['ymax']))]
        result.append((mask_status, bbox))

    return result


def extract_faces():

    for extension in ['*.png', '*.jpg', '*.tif']:
        for image_path in glob.glob(os.path.join(images_path, extension)):

            _, image_name = os.path.split(image_path)

            annotation_filename = image_name[:-4] + '.xml'

            annotation_filepath = os.path.join(annotations_path, annotation_filename)

            image = cv2.imread(image_path)

            if image is None:
                continue

            image_objects = read_annotation_file(annotation_filepath)
            count_mask = 0
            count_wrong_mask = 0
            count_no_mask = 0

            for face_object in image_objects:
                mask_status, bbox = face_object

                new_image = image[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]

                # Resizing face_objects if greater than (224, 224)
                if new_image.shape[0] > 224 or new_image.shape[1] > 224:
                    new_image = cv2.resize(new_image, (224, 224))

                resized_image = np.zeros((224, 224, 3), np.uint8)

                # Face at image center is smaller than (224, 224)
                h, w = new_image.shape[:2]
                yoff = round((224 - h) / 2)
                xoff = round((224 - w) / 2)

                resized_image[yoff:yoff + h, xoff:xoff + w] = new_image

                if mask_status == 'with_mask':
                    count_mask += 1
                    cv2.imwrite(os.path.join(mask_images, image_name[:-4]+'_0_' + str(count_mask) + '.png'), resized_image)

                elif mask_status == 'mask_weared_incorrect':
                    count_wrong_mask += 1
                    cv2.imwrite(os.path.join(incorrect_mask_images, image_name[:-4]+'_1_' + str(count_wrong_mask) + '.png'), resized_image)

                else:
                    count_no_mask += 1
                    cv2.imwrite(os.path.join(no_mask_images, image_name[:-4]+'_2_' + str(count_no_mask) + '.png'), resized_image)


if __name__ == "__main__":
    extract_faces()
