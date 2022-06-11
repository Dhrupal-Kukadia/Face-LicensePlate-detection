import darknet.darknet as dn # importing darknet wrapper for python
import cv2
import json
import os
import numpy as np


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

vehicle_network, vehicle_class_names, class_colors = dn.load_network(VEHICLE_DETECTOR_CFG, VEHICLE_DETECTOR_DATA, VEHICLE_DETECTOR_WEIGHTS)
vehicle_width, vehicle_height = dn.network_width(vehicle_network), dn.network_height(vehicle_network)
vehicle_network_shape = (vehicle_height, vehicle_width)

lp_network, lp_class_names, class_colors = dn.load_network(LP_DETECTOR_CFG, LP_DETECTOR_DATA, LP_DETECTOR_WEIGHTS)
lp_width, lp_height = dn.network_width(lp_network), dn.network_height(lp_network)
lp_network_shape = (lp_height, lp_width)

def prepare_image(network, image_path):
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
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        image = cv2.imread(image_path)

        if box:
            box = [int(item) for item in box]
            image = image[box[1]: box[3], box[0]: box[2]]

        cv2.imwrite(os.path.join(save_path, image_name), image)

def scaling(image_shape, network_shape, box):
    scale = np.divide(image_shape, network_shape)

    box = list(box)

    box[0] = scale[1] * box[0]
    box[1] = scale[0] * box[1]
    box[2] = scale[1] * box[2]
    box[3] = scale[0] * box[3]

    return box

def reset_crop(crop_location, bounding_box):
    bounding_box[0] = crop_location[0] + bounding_box[0]
    bounding_box[1] = crop_location[1] + bounding_box[1]
    bounding_box[2] = crop_location[0] + bounding_box[2]
    bounding_box[3] = crop_location[1] + bounding_box[3]

    return bounding_box

def vehicle_detector(image_path):
    image, image_shape = prepare_image(vehicle_network, image_path)
    detections = dn.detect_image(vehicle_network, vehicle_class_names, image, thresh=VEHICLE_DETECTOR_THRESHOLD)

    if(len(detections)):
        bounding_boxes = [dn.bbox2points(item[2]) for item in detections]

        scaled_bounding_boxes = [scaling(image_shape, vehicle_network_shape, box) for box in bounding_boxes]

        # result = {}

        image_name = image_path.split('/')[-1]
        save_path = os.path.splitext(image_path)[0]
        result_name = os.path.splitext(image_name)[0] + ".json"

        # filename = os.path.join("data/vehicle-detections", result_name)

        # with open(filename, mode='w', encoding='utf-8') as f:
        #     pass
        
        # with open(filename, mode="a", encoding="utf-8") as f:
        #     for i in range(len(detections)):
        #         item = {
        #             'vehicle': detections[i][0],
        #             'confidence': detections[i][1],
        #             'bounding box': scaled_bounding_boxes[i],
        #         }

        #         result[i] = item 

        #     f.write(json.dumps(result))
        #     f.close()

        print("Found " + str(len(detections)) + " vehicles in the image.")
        
        vehicles = []

        for i in range(len(scaled_bounding_boxes)):
            vehicle_name = os.path.splitext(image_name)[0] + "_" + str(i) +".png"
            vehicles.append(os.path.join(save_path, vehicle_name))
            save_image(image_path, save_path=save_path, image_name=vehicle_name, box = scaled_bounding_boxes[i])
        
        return vehicles, result_name, scaled_bounding_boxes, save_path
    
    else:
        print("No vehicles found")

def lp_detector(vehicles, result_name, crop_locations, save_path, annot_path):
    if(len(vehicles)):
        result = {}

        for i in range(len(vehicles)):
            lp_image, lp_image_shape = prepare_image(lp_network, vehicles[i])
            lp_detections = dn.detect_image(lp_network, lp_class_names, lp_image, thresh=LP_THRESHOLD)

            if(len(lp_detections)):
                bounding_boxes = [dn.bbox2points(item[2]) for item in lp_detections]

                scaled_bounding_boxes = [scaling(lp_image_shape, lp_network_shape, box) for box in bounding_boxes]

                original_bounding_boxes = [reset_crop(crop_locations[i], box) for box in scaled_bounding_boxes]


                filename = os.path.join(annot_path, result_name)

                with open(filename, mode="w", encoding="utf-8") as f:
                    pass

                with open(filename, mode="a", encoding="utf-8") as f:
                    vehicle = {}
                    for j in range(len(lp_detections)):
                        lp = {
                        'confidence': lp_detections[j][1],
                        'bounding box': original_bounding_boxes[j],
                        }  

                        vehicle[j] = lp
                    
                    result[i] = vehicle

                    f.write(json.dumps(result))
                    f.close()
                
                os.remove(vehicles[i])

                print(str(len(lp_detections)) + " license plates were found.")
            
            else:
                print("No license plates were found.")
            
        os.rmdir(save_path)

        return original_bounding_boxes

def detect(image_path, annot_path):
    vehicles, result_name, crop_locations, save_path = vehicle_detector(image_path)
    return lp_detector(vehicles, result_name, crop_locations, save_path, annot_path)