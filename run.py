"""
The file will be detecting and blurring the faces and the license plates for all the frames in a folder.
"""

from detect import detect, change_vehicle_threshold, change_lp_threshold
from tracking import finding_missing_detections
from boxes import blurring
from PIL import Image, ImageFilter
from retinaface import RetinaFace
from matplotlib import pyplot as plt
from centroidtracker import CentroidTracker
import json
import os 
import cv2 
import pathlib
import time
import numpy as np
import argparse

#NMS for best bounding box
def non_max_suppression_fast(boxes, overlapThresh):
    """
    Function implementing non max suppression for reducing number of license plates detected in a 
    single vehicle.

    Parameters:
        boxes: Bounding boxes
        overlapThresh: Threshold for overlapping
    
    Return:
        Single bounding box after NMS
    """
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))

#Blur the detections
def blur_image(blur, x1, y1, x2, y2):
    """
    Function to blur detected objects of image given its bounding boxes

    Parameters:
        blur: Image instance 
        x1, y1, x2, y2 : Bounding box coordinates

    Return:
        None, blurs the bbox patch and pastes it onto the image instance
    """
    face = blur.crop((x1, y1, x2, y2))
    face_blur = face.filter(ImageFilter.GaussianBlur(radius=20))
    blur.paste(face_blur, (x1, y1, x2, y2))

#Main script
def run(dataset, visualize):
    """
    Main script for object detection
    """
    start  = time.time()

    #Object tracking
    tracker = CentroidTracker(maxDisappeared=20, maxDistance=90)

    #Path
    root = os.path.join(os.getcwd(), dataset)

    #Iterate through root directory
    for dir in os.listdir(root):

        #Current directory
        currdir = os.path.join(root, dir)

        #Create privacy folder
        privacy_folder_path = os.path.join(currdir, "privacy")
        if not os.path.exists(privacy_folder_path):
            os.mkdir(privacy_folder_path)

        #Camera folder
        camera = os.path.join(currdir, "camera")

        #Create folder to store annotations
        annotations = os.path.join(currdir, "annotations")
        if not os.path.exists(annotations):
            os.mkdir(annotations)
        
        #Create folders for plates and faces
        license_path = os.path.join(currdir, os.path.join("annotations", "license-plates"))
        if not os.path.exists(license_path):
            os.mkdir(license_path)

        faces_path = os.path.join(currdir, os.path.join("annotations", "faces"))
        if not os.path.exists(faces_path):
            os.mkdir(faces_path) 

        #Iterate through 'cam' directories
        for cam in os.listdir(camera):
            currcam = os.path.join(camera, cam)

            #Create directory to store blurred images
            blur_folder_path = os.path.join(privacy_folder_path, cam)
            if not os.path.exists(blur_folder_path):
                os.mkdir(blur_folder_path)

            #Create directories to store annotations
            lic_path = os.path.join(license_path, cam)
            if not os.path.exists(lic_path):
                os.mkdir(lic_path)

            face_path = os.path.join(faces_path, cam)
            if not os.path.exists(face_path):
                os.mkdir(face_path)

            for frame in os.listdir(currcam):
                #Image path
                frame_path = os.path.join(currcam, frame)
                file_name = pathlib.Path(frame_path).stem

                #Open the image
                if os.path.isfile(frame_path):
                    dt = cv2.imread(frame_path)
                    blur = Image.open(frame_path)

                #License plate detection
                plates = detect(frame_path, lic_path)

                #Face detection
                faces = RetinaFace.detect_faces(frame_path, threshold=0.5) 

                #Face annotations
                face_result = {}
                for face in faces:
                    #Faces detected
                    if(len(face) != 0):
                    
                        #If detected
                        detection = faces[face]

                        #Coordinates
                        facial_area = detection['facial_area']
                        x1, y1 = facial_area[0], facial_area[1]
                        x2, y2 = facial_area[2], facial_area[3]

                        #Coordinates of face
                        cv2.rectangle(dt, (x1, y1), (x2, y2), (255, 0, 0), 2)

                        #Blur the face
                        blur_image(blur, x1, y1, x2, y2)

                        #Annotations
                        face_result[face] = {
                            'Confidence': faces[face]['score'],
                            'Bounding box': [int(point) for point in faces[face]['facial_area']]
                        } 
                        
                    with open(f"{face_path}/{file_name}.json", "w") as f:
                        f.write(json.dumps(face_result))

                    #License plate annotations
                    lic_result = {}
                    if plates is not None: 

                        #License plates
                        for vehicles in plates:
                            boxes = []
                            confidence = 0
                            for detections in plates[vehicles]:
                                for bbox in plates[vehicles][detections]:
                                    if bbox != "confidence":
                                        boxes.append(plates[vehicles][detections][bbox])
                                    else:
                                        confidence = max(confidence, float(plates[vehicles][detections][bbox])) 

                            #NMS
                            boundingboxes = np.array(boxes)
                            boundingboxes = boundingboxes.astype(int)
                            box = non_max_suppression_fast(boundingboxes, 0.3)

                            objects = tracker.update(box)
                            for (objectId, bbox) in objects.items():
                                x1, y1, x2, y2 = bbox
                                x1 = int(x1)
                                y1 = int(y1)
                                x2 = int(x2)
                                y2 = int(y2)

                                #Coordinates of plate
                                cv2.rectangle(dt, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                text = "ID: {}".format(objectId)
                                cv2.putText(dt, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

                                #Blur the plate
                                blur_image(blur, x1, y1, x2, y2)

                                #Annotations
                                lic_result[vehicles] = {
                                    objectId: {
                                        'Confidence': confidence,
                                        'Bounding box': box.tolist(),
                                    }
                                }

                    with open(f"{lic_path}/{file_name}.json", "w") as f:
                        f.write(json.dumps(lic_result))

                    if visualize:
                        # Visualization
                        plt.figure(figsize=(20, 20))
                        plt.imshow(dt[:, :, ::-1])
                        plt.show()
                    else:
                        # Save the image
                        blur.save(f"{blur_folder_path}\{file_name}.png")

                print(f"{frame} has been processed")

            print(f"Detection for {cam} of {dir} has been completed")
        
        print(f"Detection for {dir} has been completed")

    #Training/testing time
    end = time.time()
    duration = int(end - start)
    print(f"Elapsed time : {duration//60} minutes {duration%60} seconds.") # Total elapsed time in whole operation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for blurring faces and license plates")
    parser.add_argument("--vehicle_threshold", type=float, help="Change vehicle detection threshold")
    parser.add_argument("--lp_threshold", type=float, help="Change lp detection threshold")

    parser.add_argument("--visualize", help="Visualize results of detections", action="store_true")
    parser.add_argument("--data", help="Name of dataset folder")

    args = parser.parse_args()

    if args.vehicle_threshold is not None :
        change_vehicle_threshold(args.vehicle_threshold)
    
    if args.lp_threshold is not None:
        change_lp_threshold(args.lp_threshold)

    if args.visualize:
        visualize = True
    else:
        visualize = False
    
    run(args.data, visualize)

    #Postprocessing
    finding_missing_detections(args.data)

    #Blur the new detections after postprocessing
    blurring(args.data)
