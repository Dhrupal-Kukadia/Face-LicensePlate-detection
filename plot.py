import cv2
import json 
import darknet.darknet as dn
import os

def plot_image(image_path, box=None):
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        print(image.shape)

        if box :
            left, top, right, bottom = dn.bbox2points(box)
            cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)


        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(image_path + " doesn't exist.")
        

plot_image("data/sample-image.jpg", box=[
                158.466552734375,
                364.115478515625,
                169.09637451171875,
                47.05010986328125
            ])