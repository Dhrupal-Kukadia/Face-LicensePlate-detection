"""
This file contains functions to view an image, with a rectangle on detected vehicle or license plate,
if its bounding box is given.
"""
import cv2
import json 
import darknet.darknet as dn
import os

def plot_image(image_path, box=None):
    """
    Function to show image.

    Parameter:
    image_path: Path of image.
    box: Bounding box for rectangle.

    Return:
    None, opens a pop-up window showing the image.
    """
    if os.path.exists(image_path): # Checking if an image is present in the given path
        image = cv2.imread(image_path)
        print(image.shape)

        if box : # If box i given, draw it on the image
            left, top, right, bottom = [int(item) for item in box]
            cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 1)


        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(image_path + " doesn't exist.")