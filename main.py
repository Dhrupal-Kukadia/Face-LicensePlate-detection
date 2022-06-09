import darknet.darknet as dn # importing darknet wrapper for python
import cv2
import json
import os

import string
import random


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

VEHICLE_DETECTOR_CFG = "data/vehicle-detection.cfg"
VEHICLE_DETECTOR_DATA = "data/vehicle-detection.data"
VEHICLE_DETECTOR_NAMES = "data/vehicle-detection.names"
VEHICLE_DETECTOR_WEIGHTS = "data/vehicle-detection.weights"
VEHICLE_DETECTOR_THRESHOLD = 0.25

LP_DETECTOR_CFG = "data/lp-detection-layout-classification.cfg"
LP_DETECTOR_DATA = "data/lp-detection-layout-classification.data"
LP_DETECTOR_NAMES = "data/lp-detection-layout-classification.names"
LP_DETECTOR_WEIGHTS = "data/lp-detection-layout-classification.weights"
LP_THRESHOLD = 0.01

def prepare_image(network, image_path, bounding_box=None, save_path = None):
    width = dn.network_width(network)
    height = dn.network_height(network)
    darknet_image = dn.make_image(width, height, 3)

    image = cv2.imread(image_path)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    dn.copy_image_from_bytes(darknet_image, image_resized.tobytes())

    return darknet_image

def save_image(image_path, save_path, image_name, box= None):
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        image = cv2.imread(image_path)

        if box:
            image = image[box[1]: box[3], box[0]: box[2]]

        cv2.imwrite(os.path.join(save_path, image_name), image)


def vehicle_detector(image_path):
    try:
        network, class_names, class_colors = dn.load_network(VEHICLE_DETECTOR_CFG, VEHICLE_DETECTOR_DATA, VEHICLE_DETECTOR_WEIGHTS)
        width, height = dn.network_width(network), dn.network_height(network)
        print(width, height)
        image = prepare_image(network, image_path)
        detections = dn.detect_image(network, class_names, image, thresh=VEHICLE_DETECTOR_THRESHOLD)

        if(len(detections)):
            result = {}

            image_name = image_path.split('/')[-1]
            result_name = os.path.splitext(image_name)[0] + ".json"

            filename = os.path.join("data/vehicle-detections", result_name)

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

            bounding_boxes = [dn.bbox2points(item[2]) for item in detections]

            print("Found " + str(len(detections)) + " in the image.")
            
            vehicles = []

            for i in range(len(bounding_boxes)):
                vehicle_name = os.path.splitext(image_name)[0] + "_" + str(i) +".png"
                vehicles.append(os.path.join("data/vehicle-detections/images", vehicle_name))
                save_image(image_path, save_path="data/vehicle-detections/images", image_name=vehicle_name, box = bounding_boxes[i])

            image_copy = cv2.imread(image_path)

            image_copy = cv2.resize(image_copy, (width, height))

            image_copy = dn.draw_boxes(detections, image_copy, class_colors)

            cv2.imshow("Image", image_copy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            return vehicles, result_name
        
        else:
            print("No vehicles found")

    except Exception as e:
        print(e)

def lp_detector(vehicles, result_name):
    try:
        if(len(vehicles)):
            network, class_names, class_colors = dn.load_network(LP_DETECTOR_CFG, LP_DETECTOR_DATA, LP_DETECTOR_WEIGHTS)
            lp_width, lp_height = dn.network_width(network), dn.network_height(network)
            print(lp_width, lp_height)
            result = {}

            for i in range(len(vehicles)):
                lp_image = prepare_image(network, vehicles[i])
                lp_detections = dn.detect_image(network, class_names, lp_image, thresh=LP_THRESHOLD)

                if(len(lp_detections)):
                    filename = os.path.join("data/lp-detections", result_name)

                    with open(filename, mode="w", encoding="utf-8") as f:
                        pass

                    with open(filename, mode="a", encoding="utf-8") as f:
                        vehicle = {}
                        for j in range(len(lp_detections)):
                            lp = {
                            'confidence': lp_detections[j][1],
                            'bounding box': list(lp_detections[j][2]),
                            }  

                            vehicle[j] = lp
                        
                        result[i] = vehicle

                        f.write(json.dumps(result))
                        f.close()
                    
                    print(str(len(lp_detections)) + " license plates were found.")

                    lp_image_copy = cv2.imread(vehicles[i])

                    lp_image_copy = cv2.resize(lp_image_copy, (lp_width, lp_height))

                    lp_image_copy = dn.draw_boxes(lp_detections, lp_image_copy, class_colors)

                    cv2.imshow("Image", lp_image_copy)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                
                else:
                    print("No license plates were found.")

    except Exception as e:
        print(e)

def detect(image_path):
    vehicles, result_name = vehicle_detector(image_path)
    lp_detector(vehicles, result_name)

detect("data/sample-image.jpg")