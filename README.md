Instructions to run the project:
1. Download (https://www.kaggle.com/andrewmvd/face-mask-detection) the face mask dataset.
2. Run Data_Preparation.ipynb file to create the subfolders to store face crop images based on classes. This script also resizes the images to (224, 224). Change the paths wherever necessary.
3. To train the model, run mask-detect_Sundaresh.ipynb. Change learning rate, number of epochs, batch size and optimizer according to the requirements and provide the correct path to save the model.
4. Inference.ipynb can be used to perform inference on a single image or a folder of images. The results are stored in the specified destination folder. Provide the path to trained classification model and yolov3 face detection model.
5. Run Face_mask_classification_metrics.ipynb to generate classification metrics of the trained model on test dataset.
