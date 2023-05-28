import numpy as np
import pgmpy
import os
from collections import namedtuple
import cv2 as cv
import argparse
from pathlib import Path

Detection = namedtuple("Detection", ["image_path", "num", "Bbox"])


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def load_bboxes(image_dir, image_path):
    detections = []
    with open("bboxes.txt", "r") as file:
        lines = file.readlines()
        line_index = 0
        while line_index < len(lines):
            current_image_name = lines[line_index].strip()
            curr_image_dir = os.path.join(image_dir, current_image_name)
            if str(curr_image_dir) == str(image_path):
                num = int(lines[line_index + 1])                 # Bboxes number
                Bboxes = []                                      # empty Bbox list
                for _ in range(num):
                    bbox_str = lines[line_index + 2]
                    bbox = list(map(float, bbox_str.strip().split()))
                    Bboxes.append(bbox)
                    line_index += 1
                detections.append(Detection(image_path, num, Bboxes))
                break
            else:
                line_index += 1
    return detections
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str)
    args = parser.parse_args()
    images_dir = Path(args.images_dir)
    images_paths = sorted([image_path for image_path in images_dir.iterdir() if image_path.name.endswith('.jpg')])

    for image_path in images_paths:
        img_boxes = load_bboxes(images_dir,image_path)
        print(img_boxes)
        print(image_path)
        #iou = bb_intersection_over_union(Detection.gt, detection.pred)

        print()
        image = cv.imread(str(image_path))
        if image is None:
            print(f'Error loading image {image_path}')
            continue
        cv.imshow('image', image)
        key = cv.waitKey(0)

        if key == 27:
            break



if __name__ == '__main__':
    main()