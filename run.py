"""
The file will be detecting and blurring the faces and the license plates for all the frames in a folder.
"""

from detect import detect
from PIL import Image, ImageFilter
from retinaface import RetinaFace
from matplotlib import pyplot as plt
import json
import os 
import cv2 
import pathlib
import time

def blur_image(blur, x1, y1, x2, y2):
    #Face blur
    face = blur.crop((x1, y1, x2, y2))
    face_blur = face.filter(ImageFilter.GaussianBlur(radius=20))
    blur.paste(face_blur, (x1, y1, x2, y2))

start  = time.time()

#Path
root = os.path.join(os.getcwd(), "IDD2_Subset")

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

    visualize = True

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
            dt = cv2.imread(frame_path)
            blur = Image.open(frame_path)

            #License plate detection
            plates = detect(frame_path, lic_path)

            #Face detection
            faces = RetinaFace.detect_faces(frame_path, threshold=0.5) 

            print(f"{len(faces)} faces were detected in {frame}")

            result = {}

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
                    result[face] = {
                        'Confidence': faces[face]['score'],
                        'Bounding box': [int(point) for point in faces[face]['facial_area']]
                    } 
                    
                with open(f"{face_path}/{file_name}.json", "w") as f:
                    f.write(json.dumps(result))

                #License plates detected
                if plates is not None: 
                    print(f"{len(plates)} license plates were detected in {frame}") 
                    for box in plates:
                        left, top, right, bottom = [int(item) for item in box] # Converting bounding box coordinates to integer type

                        #Coordinates of plate
                        cv2.rectangle(dt, (left, top), (right, bottom), (255, 0, 0), 2) 

                        #Blur the plate
                        blur_image(blur, left, top, right, bottom)

                if visualize:
                    #Visualization
                    plt.figure(figsize=(20, 20))
                    plt.imshow(dt[:, :, ::-1])
                    plt.show()
                else:
                    #Save the image
                    blur.save(f"{blur_folder_path}\{file_name}.png")

        print(f"Detection for {cam} of {dir} has been completed")
    
    print(f"Detection for {dir} has been completed")

#Training/testing time
end = time.time()
duration = int(end - start)
print(f"Elapsed time : {duration//60} minutes {duration%60} seconds.") # Total elapsed time in whole operation