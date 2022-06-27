import os
import cv2
import json

#Path
root = os.path.join(os.getcwd(), "")

#Iterate through root directory
for dir in os.listdir(root):

    #Current directory
    currdir = os.path.join(root, dir)

    #Camera directory
    camera = os.path.join(currdir, "camera")

    #Annotations directory
    annotations = os.path.join(currdir, "annotations")

    #Create directory for storing the images
    postprocessing = os.path.join(currdir, "postprocessing")
    if not os.path.exists(postprocessing):
        os.mkdir(postprocessing)

    #License plates directory
    license_path = os.path.join(annotations, "license-plates")

    for cam in os.listdir(license_path):

        #Create directory to store blurred images for each cam
        blur_folder_path = os.path.join(postprocessing, cam)
        if not os.path.exists(blur_folder_path):
            os.mkdir(blur_folder_path)

        currcam = os.path.join(license_path, cam)
        cam_dir = os.path.join(camera, cam)

        for file in os.listdir(currcam):

            #Current 'json' file
            currfile = os.path.join(currcam, file)

            #Fetch the image from camera directory
            img_name = file[0:5] +".png"
            img_path = os.path.join(cam_dir, img_name)
            img = cv2.imread(img_path)

            #Open the 'json' file
            with open(currfile, 'r') as f:
                detections = json.loads(f.read())

            #Use the annotations for bounding boxes
            if len(detections) != 0:
                for vehicles in detections:
                    for objectID in detections[vehicles]:
                        x1, y1, x2, y2 = [element for innerList in detections[vehicles][objectID]["Bounding box"] for element in innerList]   
                        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2.imwrite(f"{blur_folder_path}\{img_name}", img)