"""
The file will detect all the frames in a folder, detect and blur all the license plates present in all the vehicles 
present in them.
"""

from detect import detect
import os 
import cv2 
from PIL import Image, ImageFilter
import time

start  = time.time()

root = os.path.join(os.getcwd(), "IDD2_Subset") # Get directory of root directory

for dir in os.listdir(root): # Iterating through all the sequences in the root directory

    currdir = os.path.join(root, dir) # Sequence directory
    privacy_path = os.path.join(currdir, "privacy") # Directory of privacy folder in the sequence

    annotations_path = os.path.join(currdir, os.path.join("annotations", "license-plates")) # Path where annotations should be saved

    for cam in os.listdir(privacy_path): # Iterating through different camera frames in privacy folder
        currcam = os.path.join(privacy_path, cam) # Camera directory
        annot_path = os.path.join(annotations_path, cam) # Annotations file location

        for frame in os.listdir(currcam): # Iterating through all files in camera directory
            frame_path = os.path.join(currcam, frame) # Path of frame

            if os.path.isfile(frame_path): # Checking whether path is a file or not
                dt = cv2.imread(frame_path) # Creating two instances of frame
                blur = Image.open(frame_path)

                boxes = detect(frame_path, annot_path) # Getting bounding boxes of license plates in the frame

                if boxes is not None: # If there are license plates in the frame
                    print(f"{len(boxes)} license plates were found in {frame}") 
                    for box in boxes:
                        left, top, right, bottom = [int(item) for item in box] # Converting bounding box coordinates to integer type

                        # Drawing rectangle on license plate, cropping it from one instance and pasting it on another
                        cv2.rectangle(dt, (left, top), (right, bottom), (255, 0, 0), 2) 
                        face = blur.crop((left, top, right, bottom))
                        face_blur = face.filter(ImageFilter.GaussianBlur(radius=20))
                        blur.paste(face_blur, (left, top, right, bottom)) 

                        blur.save(frame_path) # Saving resulting image
            
        print(f"Detection for {cam} of {dir} has been completed")
    
    print(f"Detection for {dir} has been completed")

end = time.time()

duration = int(end - start)

print(f"Elapsed time : {duration//60} minutes {duration%60} seconds.") # Total elapsed time in whole operation

