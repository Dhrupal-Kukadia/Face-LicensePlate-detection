"""
This module has functions to delete unnecessary files for ablation study and produce results for it.
"""
from detect import detect
import os 
import cv2 
from PIL import Image, ImageFilter
import time
import sys

start = time.time()

root = os.path.join(os.getcwd(), "ablation-study")

def delete():
    """
    Function for deleting frames which are not required for ablation study
    """
    for dir in os.listdir(root):

        #Current directory
        currdir = os.path.join(root, dir)
        
        #Create privacy folder
        privacy = os.path.join(currdir, "privacy")
        
        #Camera folder
        camera = os.path.join(currdir, "camera")

        #Iterate through 'cam' directories
        for cam in os.listdir(camera):
            currcam = os.path.join(camera, cam)

            for i, frame in enumerate(os.listdir(currcam)):
                if i % 5 != 0: # Only keeping every fifth frame in a sequence
                    os.remove(os.path.join(currcam, frame)) # Deleting frame from camera folder

        for cam in os.listdir(privacy):
            currcam = os.path.join(privacy, cam)

            for i, frame in enumerate(os.listdir(currcam)):
                if i % 5 != 0: 
                    os.remove(os.path.join(currcam, frame)) # Deleting frame from privacy folder

    print(f"Total time elapsed : {int(time.time() - start)//60} minutes {int(time.time() - start)%60} seconds.")

def ablation_study(path):
    """
    Function to generate images for ablation study a given path.

    Parameters:
        path: name of folder in which images are to be kept.
    """
    for dir in os.listdir(root): # Iterating through all the sequences in the root directory

        currdir = os.path.join(root, dir) # Sequence directory
        privacy_path = os.path.join(currdir, "privacy") # Directory of privacy folder in the sequence
        
        if os.path.isdir(currdir):
            ablation_path = os.path.join(currdir, path) # Directory for each study
        else:
            break
        if not os.path.exists(ablation_path):
            os.mkdir(ablation_path)

        annotations_path = os.path.join(currdir, os.path.join(os.path.join("annotations", path), "license-plates")) # Path where annotations should be saved

        for cam in os.listdir(privacy_path): # Iterating through different camera frames in privacy folder
            currcam = os.path.join(privacy_path, cam) # Camera directory
            annot_path = os.path.join(annotations_path, cam) # Annotations file location
            
            study_path = os.path.join(ablation_path, cam) # 'cam' directory inside study_path
            if not os.path.exists(study_path):
                os.mkdir(study_path) 

            for frame in os.listdir(currcam): # Iterating through all files in camera directory
                frame_path = os.path.join(currcam, frame) # Path of frame
                save_path = os.path.join(study_path, frame) # Path of image for ablation study

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

                            blur.save(save_path) # Saving resulting image
                
            print(f"Detection for {cam} of {dir} has been completed")
        
        print(f"Detection for {dir} has been completed")

    print(f"Total time elapsed : {int(time.time() - start)//60} minutes {int(time.time() - start)%60} seconds.") # Total elapsed time in whole operation


if __name__ == "__main__":
    if sys.argv[1] == "delete": # For delete operation
        delete()
    else: # For producing images for ablation study inside the directory of the given name
        path = sys.argv[1]
        ablation_study(path)