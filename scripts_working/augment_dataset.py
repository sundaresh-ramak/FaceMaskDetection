import os
import numpy as np
import cv2
import glob

incorrect_mask_images = "D:/Projects/FaceMaskClassification/data/wrong_mask"
no_mask_images = "D:/Projects/FaceMaskClassification/data/no_mask"

wrong_mask_aug = "D:/Projects/FaceMaskClassification/data/wrong_mask_aug"
no_mask_aug = "D:/Projects/FaceMaskClassification/data/no_mask_aug"

if not os.path.exists(wrong_mask_aug):
    os.mkdir(wrong_mask_aug)

if not os.path.exists(no_mask_aug):
    os.mkdir(no_mask_aug)


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def flip_image(image):
    return cv2.flip(image, 1)


for images_path in [incorrect_mask_images, no_mask_images]:
    for image_path in glob.glob(os.path.join(images_path, '*.png')):
        orig_image = cv2.imread(image_path)
        head, tail = os.path.split(image_path)

        rotated_20p = rotate_image(orig_image, 20)
        rotated_20n = rotate_image(orig_image, -20)
        flipped = flip_image(orig_image)
        destination_path_20n = None  # folder path for images rotated by -20 deg
        destination_path_20p = None  # folder path for images rotated by +20 deg
        destination_path_flip = None  # folder path for horizontally flipped images
        if images_path == incorrect_mask_images:
            destination_path_20p = os.path.join(wrong_mask_aug, tail[:-4]+'_20p.png')
            destination_path_20n = os.path.join(wrong_mask_aug, tail[:-4]+'_20n.png')
            destination_path_flip = os.path.join(wrong_mask_aug, tail[:-4]+'_flip.png')

            cv2.imwrite(destination_path_20p, rotated_20p)
            cv2.imwrite(destination_path_20n, rotated_20n)
            cv2.imwrite(destination_path_flip, flipped)

        if images_path == no_mask_images:
            destination_path_20p = os.path.join(no_mask_aug, tail[:-4]+'_20p.png')
            destination_path_20n = os.path.join(no_mask_aug, tail[:-4]+'_20n.png')
            destination_path_flip = os.path.join(no_mask_aug, tail[:-4]+'_flip.png')

            cv2.imwrite(destination_path_20p, rotated_20p)
            cv2.imwrite(destination_path_20n, rotated_20n)
            cv2.imwrite(destination_path_flip, flipped)
