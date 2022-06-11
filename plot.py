import cv2
import json 
import darknet.darknet as dn
import os

def plot_image(image_path, box=None):
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        print(image.shape)

        if box :
            left, top, right, bottom = [int(item) for item in box]
            cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 1)


        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(image_path + " doesn't exist.")
        

plot_image("data/679660_leftImg8bit.png", box= [
                502.914979757085,
                438.14903846153845,
                560.6072874493927,
                452.1129807692308
            ])