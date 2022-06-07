from email.mime import image
from fileinput import filename
import darknet.darknet as dn # importing darknet wrapper for python
import cv2
import json
import os
import re

VEHICLE_DETECTOR_CFG = "data/vehicle-detection.cfg"
VEHICLE_DETECTOR_DATA = "data/vehicle-detection.data"
VEHICLE_DETECTOR_NAMES = "data/vehicle-detection.names"
VEHICLE_DETECTOR_WEIGHTS = "data/vehicle-detection.weights"
VEHICLE_DETECTOR_THRESHOLD = 0.25

def prepare_image(network, image_path, bounding_box=None):
    width = dn.network_width(network)
    height = dn.network_height(network)
    darknet_image = dn.make_image(width, height, 3)

    if bounding_box is None:
        image = cv2.imread(image_path)
    else:
        x1, y1, x2, y2 = dn.bbox2points(bounding_box)
        image = cv2.imread(image_path)[x1: x2, y1: y2]

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    dn.copy_image_from_bytes(darknet_image, image_resized.tobytes())

    return darknet_image

def vehicle_detector(image_path):
    try:
        network, class_names, class_colors = dn.load_network(VEHICLE_DETECTOR_CFG, VEHICLE_DETECTOR_DATA, VEHICLE_DETECTOR_WEIGHTS)
        image = prepare_image(network, image_path)
        detections = dn.detect_image(network, class_names, image, thresh=VEHICLE_DETECTOR_THRESHOLD)

        result = {}

        image_name = image_path.split('\\')[-1]
        result_name = os.path.splitext(image_name)[0] + ".json"

        filename = os.path.join("data/vehicle-detections", result_name)
        print(filename)

        with open(filename, mode='w', encoding='utf-8') as f:
            pass
        
        with open(filename, mode="a", encoding="utf-8") as f:
            for i in range(len(detections)):
                item = {
                    'vehicle': detections[i][0],
                    'confidence': detections[i][1],
                    'bounding box': list(detections[i][2]),
                }

                result[i] = item 

            f.write(json.dumps(result))
            f.close()

        bounding_boxes = [item[2] for item in detections]

        vehicles = [prepare_image(network, image_path, box) for box in bounding_boxes]

        return vehicles

    except Exception as e:
        print(e)
