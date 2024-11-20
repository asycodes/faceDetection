################################################
#
# COMS30068 - face.py
# University of Bristol
#
################################################

import numpy as np
import cv2
import os
import sys
import argparse

# LOADING THE IMAGE
# Example usage: python filter2d.py -n car1.png
parser = argparse.ArgumentParser(description='face detection')
parser.add_argument('-name', '-n', type=str, default='images/face1.jpg')
args = parser.parse_args()

# /** Global variables */
cascade_name = "frontalface.xml"
ground_truth_bounding_boxes = [] # to store all boundary boxes for IOU computation
pred_bounding_boxes = []

def detectAndDisplay(frame):

	# 1. Prepare Image by turning it into Grayscale and normalising lighting
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    cv2.imwrite( "preprocessed.jpg", frame_gray )
    # 2. Perform Viola-Jones Object Detection
    faces = model.detectMultiScale(frame_gray, scaleFactor=1.07, minNeighbors=2, flags=0, minSize=(10,10), maxSize=(300,300))
    readGroundtruth(frame)
    # 4. Draw box around faces found
    for i in range(0, len(faces)):
        start_point = (faces[i][0], faces[i][1])
        end_point = (faces[i][0] + faces[i][2], faces[i][1] + faces[i][3])
        pred_bounding_boxes.append((start_point,end_point,faces[i][2],faces[i][3]))
        colour = (0, 255, 0)
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)
    success = evaluate_iou()
    tpr_result = tpr(success)
    f1_result = f1(success)
    print("SUCCESS: ",success)
    print("TPR: ",tpr_result)
    print("F1: ",f1_result)



# ************ NEED MODIFICATION ************
def readGroundtruth(frame,filename='groundtruth.txt'):
    # read bounding boxes as ground truth
    with open(filename) as f:
        # read each line in text file
        for line in f.readlines():
            content_list = line.split(",")
            img_name = content_list[0]
            if img_name == imageName.lstrip("images/").rstrip(".jpg"):
                x = int(float(content_list[1]))
                y = int(float(content_list[2]))
                width = int(float(content_list[3]))
                height = int(float(content_list[4]))
                print(str(x)+' '+str(y)+' '+str(width)+' '+str(height))
                ## draw
                start_point = (x, y)
                end_point = (x + width, y + height)
                ground_truth_bounding_boxes.append((start_point,end_point,width,height))
                colour = (0, 0, 255)
                thickness = 2
                frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)


def iou(pred,truth):
    # coord of intersection points
    start_x = max(pred[0][0],truth[0][0])
    start_y = max(pred[0][1],truth[0][1])
    end_x = min(pred[1][0],truth[1][0])
    end_y = min(pred[1][1],truth[1][1])

    area_inters = max(0,(end_x - start_x)) * max(0,(end_y-start_y))
    area_pred = (pred[1][0]-pred[0][0]) * (pred[1][1]-pred[0][1])
    area_truth = (truth[1][0]-truth[0][0]) * (truth[1][1]-truth[0][1])
    area_union = area_pred+area_truth - area_inters

    iou_score = area_inters/area_union
    return iou_score



def evaluate_iou():
    # loop each pred with each ground, if over threshold we +1 success
    success = 0
    for pred in pred_bounding_boxes:
        for truth in ground_truth_bounding_boxes:
            if iou(pred,truth) > 0.5: # if more than threshold, we +1 success
                success +=1
                break
    return success


def tpr(success):
    return success/len(ground_truth_bounding_boxes)

def f1(success):
    #2*[(precision*recall)/(precision+recall)]
    precision = success/len(pred_bounding_boxes)
    #false negative is truth - pred
    recall = success/( success + len(ground_truth_bounding_boxes) - success)
    return 2* ((precision*recall)/(precision+recall))



# ==== MAIN ==============================================

imageName = args.name

# ignore if no such file is present.
if (not os.path.isfile(imageName)) or (not os.path.isfile(cascade_name)):
    print('No such file')
    sys.exit(1)

# 1. Read Input Image
frame = cv2.imread(imageName, 1)

# ignore if image is not array.
if not (type(frame) is np.ndarray):
    print('Not image data')
    sys.exit(1)


# 2. Load the Strong Classifier in a structure called `Cascade'
model = cv2.CascadeClassifier()
if not model.load(cascade_name): # if got error, you might need `if not model.load(cv2.samples.findFile(cascade_name)):' instead
    print('--(!)Error loading cascade model')
    exit(0)


# 3. Detect Faces and Display Result
detectAndDisplay( frame )

# 4. Save Result Image
cv2.imwrite( "detected.jpg", frame )


