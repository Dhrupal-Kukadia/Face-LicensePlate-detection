import json
import os

def dissecting_the_frames(x1, x2, y1, y2, m, n):
    x = (float)((n * x1)+(m * x2))/(m + n) 
    y = (float)((n * y1)+(m * y2))/(m + n) 

    return [int(x), int(y)]

#Path
root = os.path.join(os.getcwd(), "")

#Iterate through root directory
for dir in os.listdir(root):

    #Current directory
    currdir = os.path.join(root, dir)

    #Annotations directory
    annotations = os.path.join(currdir, "annotations")

    #License plates directory
    license_path = os.path.join(annotations, "license-plates")

    #Iterate through 'cam' directories
    for cam in os.listdir(license_path):
        currcam = os.path.join(license_path, cam)

        #Parameters
        objID = 0
        vehicle = 0
        detectedFrames = {}
        emptyFrames = []

        #Iterate through 'json' files
        for frame in os.listdir(currcam):

            #Open the file
            frame_path = os.path.join(currcam, frame)
            with open(frame_path, 'r') as f:
                detections = json.loads(f.read())
                
                #Detections
                if len(detections) != 0:
                    for vehicles in detections:
                        for objectID in detections[vehicles]:
                            vehicle = vehicles
                            objID = objectID
                            detectedFrames[frame] = [element for innerList in detections[vehicles][objectID]["Bounding box"] for element in innerList]
                #No detections
                else:
                    if len(detectedFrames) == 1:
                        emptyFrames.append(frame)

                #Fill in the coordinates
                if len(detectedFrames) == 2:

                    #Detected frames
                    frames = []
                    for key in detectedFrames:
                        frames.append(key)

                    #Skipped frames
                    totalSkippedFrames = len(emptyFrames)
                    if totalSkippedFrames != 0 and totalSkippedFrames <= 5:

                        #Dissecting the skipped frames
                        lineSegments = totalSkippedFrames + 1
                        x11, y11, x21, y21 = detectedFrames[frames[0]]
                        x12, y12, x22, y22 = detectedFrames[frames[1]]

                        m = 1
                        print(emptyFrames)
                        for file in emptyFrames:
                            if m != lineSegments:
                                n = lineSegments - m
                                x1, y1 = dissecting_the_frames(x11, x12, y11, y12, m, n)
                                x2, y2 = dissecting_the_frames(x21, x22, y21, y22, m, n)

                                m += 1

                            #Dump the annotations
                            result = {
                                vehicle: {
                                    objID: {
                                        'Confidence': -1,
                                        'Bounding box': [[x1, y1, x2, y2]],
                                    }
                                }
                            }
                            store_path = os.path.join(currcam, file)
                            with open(f"{store_path}", "w") as f:
                                f.write(json.dumps(result))

                    #Reset the parameters
                    vehicle = 0
                    objID = 0
                    detectedFrames.pop(frames[0])
                    emptyFrames.clear()