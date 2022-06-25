"""
This file contains the code for the functions for vehicle and license plate detections, as well as some 
utility functions which are needed for doing related tasks like scaling, cropping and saving bounding box dimensions.
"""

import darknet.darknet as dn # importing darknet wrapper for python
import cv2
import os
import numpy as np
import shutil

# loading necessary files for vehicle detecting network
VEHICLE_DETECTOR_CFG = "data/vehicle-detection.cfg"
VEHICLE_DETECTOR_DATA = "data/vehicle-detection.data"
VEHICLE_DETECTOR_NAMES = "data/vehicle-detection.names"
VEHICLE_DETECTOR_WEIGHTS = "data/vehicle-detection.weights"
VEHICLE_DETECTOR_THRESHOLD = 0.25

# loading necessary files for lp detecting network
LP_DETECTOR_CFG = "data/lp-detection-layout-classification.cfg"
LP_DETECTOR_DATA = "data/lp-detection-layout-classification.data"
LP_DETECTOR_NAMES = "data/lp-detection-layout-classification.names"
LP_DETECTOR_WEIGHTS = "data/lp-detection-layout-classification.weights"
LP_THRESHOLD = 0.01

# Loading vehicle and lp detector network
vehicle_network, vehicle_class_names, class_colors = dn.load_network(VEHICLE_DETECTOR_CFG, VEHICLE_DETECTOR_DATA, VEHICLE_DETECTOR_WEIGHTS)
vehicle_width, vehicle_height = dn.network_width(vehicle_network), dn.network_height(vehicle_network)
vehicle_network_shape = (vehicle_height, vehicle_width)

lp_network, lp_class_names, class_colors = dn.load_network(LP_DETECTOR_CFG, LP_DETECTOR_DATA, LP_DETECTOR_WEIGHTS)
lp_width, lp_height = dn.network_width(lp_network), dn.network_height(lp_network)
lp_network_shape = (lp_height, lp_width)

def prepare_image(network, image_path):
    """
    This function converts image from OpenCV format to Darknet format. It rescales the image and takes
    its byte copy which is required by Darknet for prediction.

    Parameters:
        network: Loaded network object, the rescaling size is obtained from here.
        image_path: Path of the image to be converted.
    """
    width = dn.network_width(network)
    height = dn.network_height(network)
    darknet_image = dn.make_image(width, height, 3)

    image = cv2.imread(image_path)
    image_shape = image.shape[:2]

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    dn.copy_image_from_bytes(darknet_image, image_resized.tobytes())

    return darknet_image, image_shape

def save_image(image_path, save_path, image_name, box= None):
    """
    Function to save image from a location to a given location after cropping, if bounding box coordinates
    are provided.

    Parameters:
        image_path: Location of image to be saved.
        save_path: Location where image has to be saved, if it doesn't exist, it will be created.
        image_name: Name of the image file.
        box: Bounding box coordinates.
    """
    if box:
        for point in box:
            if point < 0:
                return False

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        image = cv2.imread(image_path)
        box = [int(item) for item in box]
        image = image[box[1]: box[3], box[0]: box[2]]

        cv2.imwrite(os.path.join(save_path, image_name), image)

        return True

def scaling(image_shape, network_shape, box):
    """
    Function for scaling image as per the given size. It is originally meant for rescaling the bounding box coordinates
    from the network's required resolution to its original resolution. 

    Parameters:
        image_shape: Original resolution of image.
        network_shape: Network's input resolution.
        box: Bounding box coordinates.
    """
    scale = np.divide(image_shape, network_shape)

    box = list(box)

    box[0] = scale[1] * box[0]
    box[1] = scale[0] * box[1]
    box[2] = scale[1] * box[2]
    box[3] = scale[0] * box[3]

    return box

def reset_crop(crop_location, bounding_box):
    """
    Function to obtain original coordinates of a bounding box obtained on a crop from an image.

    Parameters:
        crop_locations: Bounding box of crop, from where bounding boxes are obtained.
        bounding_box: Bounding box of detected object.
    """
    bounding_box[0] = crop_location[0] + bounding_box[0]
    bounding_box[1] = crop_location[1] + bounding_box[1]
    bounding_box[2] = crop_location[0] + bounding_box[2]
    bounding_box[3] = crop_location[1] + bounding_box[3]

    return bounding_box

def vehicle_detector(image_path, annot_path):
    """
    Function to detect vehicles in an image.

    Parameters:
        image_path: Path of image to be used for detection.

    Return:
        If no vehicles are detected, NoneType object is returned.
        Else, list of locations of cropped images of detected vehicles, name of annotation file,
        rescaled bounding boxes and save path for annotation file.
    """
    image, image_shape = prepare_image(vehicle_network, image_path)
    detections = dn.detect_image(vehicle_network, vehicle_class_names, image, thresh=VEHICLE_DETECTOR_THRESHOLD)

    image_name = image_path.split('\\')[-1]
    save_path = os.path.splitext(image_path)[0]
    result_name = os.path.splitext(image_name)[0] + ".json"

    if not os.path.exists(annot_path):
        os.makedirs(annot_path)

    filename = os.path.join(annot_path, result_name)

    with open(filename, mode="w", encoding="utf-8") as f:
        pass

    if(len(detections)):
        bounding_boxes = [dn.bbox2points(item[2]) for item in detections]

        scaled_bounding_boxes = [scaling(image_shape, vehicle_network_shape, box) for box in bounding_boxes]
        
        vehicles = []

        for i in range(len(scaled_bounding_boxes)):
            vehicle_name = image_name + "_" + str(i) +".png"
            if save_image(image_path, save_path=save_path, image_name=vehicle_name, box = scaled_bounding_boxes[i]):
                vehicles.append(os.path.join(save_path, vehicle_name))
        
        return vehicles, result_name, scaled_bounding_boxes, save_path
    
    else:
        return None

def lp_detector(vehicles, result_name, crop_locations, save_path, annot_path):
    """
    Function to detect license plates in detected vehicles' cropped images.

    Parameters:
        vehicles: List of locations where cropped images of vehicles are saved.
        result_name: File name of annotation file.
        crop_locations: Bounding box of vehicles, from where their cropped images are obtained.
        save_path: Path where cropped images are saved.
        annot_path: Path where annotation file will be saved.

    Return:
        If, no license plates are detected, empty list is returned
        Else, list of bounding box on original images of license plates.
    """

    filename = os.path.join(annot_path, result_name)

    if(len(vehicles)):
        result = {}

        for i in range(len(vehicles)):
            lp_image, lp_image_shape = prepare_image(lp_network, vehicles[i])
            lp_detections = dn.detect_image(lp_network, lp_class_names, lp_image, thresh=LP_THRESHOLD)

            if(len(lp_detections)):
                bounding_boxes = [dn.bbox2points(item[2]) for item in lp_detections]

                scaled_bounding_boxes = [scaling(lp_image_shape, lp_network_shape, box) for box in bounding_boxes]

                original_bounding_boxes = [reset_crop(crop_locations[i], box) for box in scaled_bounding_boxes]

                vehicle = {}
                for j in range(len(lp_detections)):
                    lp = {
                    'confidence': lp_detections[j][1],
                    'bounding box': original_bounding_boxes[j],
                    }  

                    vehicle[j] = lp
                    
                result[i] = vehicle            
            else:
                original_bounding_boxes = None
        
        shutil.rmtree(save_path)

        return result

def detect(image_path, annot_path):
    """
    Wrapper function for end-to-end detection of license plates.

    Parameters:
        image_path: Location of image for detection.
        annot_path: Path for saving bounding box annotation.

    Return:
        If no vehicles are found, empty list is returned.
        Else, list of bounding box on original images of license plates.
    """
    if vehicle_detector(image_path, annot_path) is None:
        return []
    else:
        vehicles, result_name, crop_locations, save_path = vehicle_detector(image_path, annot_path)
        return lp_detector(vehicles, result_name, crop_locations, save_path, annot_path)
