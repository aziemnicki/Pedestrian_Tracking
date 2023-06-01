import numpy as np
import pgmpy
import itertools
from pgmpy.models import MarkovModel
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import Mplp
import os
from collections import namedtuple
import cv2 as cv
import argparse
from pathlib import Path

Detection = namedtuple("Detection", ["image_path", "num", "Bbox"])

# metoda działa jedynie dla Bboxów blisko siebie
def bb_intersection_over_union(bbox1, bbox2):
    # determine the (x, y)-coordinates of the intersection rectangle
    x1_min, y1_min, x1_max, y1_max = bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]
    x2_min, y2_min, x2_max, y2_max = bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]

    # Oblicz wspólne obszary (intersection)
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    # compute the area of intersection rectangle
    intersection_area = max(0, x_inter_max - x_inter_min) * max(0, y_inter_max - y_inter_min)

    # Oblicz pola powierzchni obu bounding boxów
    bbox1_area = bbox1[2] * bbox1[3]
    bbox2_area = bbox2[2] * bbox2[3]

    # Oblicz IoU
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)

    return iou


def calculate_histogram(image, bbox):
    # Wyciągnij obszar z obrazu na podstawie bounding boxa
    x, y, w, h = map(int, bbox)
    roi = image[y:y+h, x:x+w]

    # Konwertuj obszar na przestrzeń barw HSV
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

    # Oblicz histogram kolorów w przestrzeni HSV
    hist = cv.calcHist([hsv_roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    # Znormalizuj histogram
    cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)

    return hist.flatten()

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

def create_factor(var_names, var_vals, params, feats, obs):

    # shape of the values array
    f_vals_shape = [len(vals) for vals in var_vals]
    # list of values, will be reshaped later
    f_vals = []
    # for all combinations of variables values
    for vals in itertools.product(*var_vals):
        # value for current combination
        cur_f_val = 0
        # for each feature
        for fi, cur_feat in enumerate(feats):
            # value of feature multipled by parameter value
            cur_f_val += params[fi] * cur_feat(*vals, obs)
        f_vals.append(np.exp(cur_f_val))
    # reshape values array
    f_vals = np.array(f_vals)
    f_vals = f_vals.reshape(f_vals_shape)

    return DiscreteFactor(var_names, f_vals_shape, f_vals)

def pairwise_feat(xi, xj, obs):
    # 1 if the same, 0 otherwise
    if xi == xj:
        val = 1
    else:
        val = 0
    return val

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str)
    args = parser.parse_args()
    images_dir = Path(args.images_dir)
    images_paths = sorted([image_path for image_path in images_dir.iterdir() if image_path.name.endswith('.jpg')])
    detection_prev = []
    hist_prev = []
    hist = []
    first_frame = True
    for image_path in images_paths:

        image = cv.imread(str(image_path))
        if image is None:
            print(f'Error loading image {image_path}')
            continue

        detection = load_bboxes(images_dir,image_path)
        hist_prev = hist.copy()
        hist = []
        results = []
        # print(detection)
        iou = []
        for box in detection[0].Bbox:
            x, y, w, h = box[0], box[1], box[2], box[3]
            # print(x, y, w, h)
            cv.rectangle(image, (int(x), int(y)), (int(x)+int(w), int(y)+int(h)), (0, 255, 0), 2)
            hist.append(calculate_histogram(image, box))            # Histogram aktualnej klatki

        if first_frame:
            # initialize previous detections and histograms
            detection_prev = detection[0].Bbox
            hist_prev.append(calculate_histogram(image, detection[0].Bbox[0]))
            img_prev = image
            first_frame = False
            continue

        for box_p in detection_prev:
            iou.append(bb_intersection_over_union(box_p, box))      # IoU między 2 klatkami

            if len(hist_prev) < len(hist):          #TODO sprawdzić ilość w liście
                hist_prev = hist.copy()
            else:
                hist_prev = hist_prev[:len(hist)]

            for i in range(len(hist)):
                result = cv.compareHist(hist_prev[i], hist[i], cv.HISTCMP_CORREL)
                results.append(result)
            hist_prev.append(calculate_histogram(img_prev, box_p))  # Histogram poprzedniej klatki

        #print(hist)
        #print(hist_prev)
        print(results)

        nodes = []  # bounding boxy
        for prev in range(len(images_paths)):
            for now in range(len(images_paths)):
                nodes.append('x_' + str(prev) + '_' + str(now))


        factors_p = []  # prawdopodobieństwo z porównania histogramów
        edges_p = []  # Krawędzie między node ami
        for prev in range(len(detection_prev)):
            for now in range(len(detection[0].Bbox)):
                cur_f_r = create_factor(['x_' + str(prev) + '_' + str(now), 'x_' + str(prev + 1) + '_' + str(now)],
                                        [[0, 1], [0, 1]],
                                        [0.5],
                                        [pairwise_feat],
                                        None)
                factors_p.append(cur_f_r)
                edges_p.append(('x_' + str(prev) + '_' + str(now), 'x_' + str(prev + 1) + '_' + str(now)))



        # factors_u = []
        # for r in range(detection[0].num):
        #         cur_f = create_factor(['x_' + str(r) + '_'],
        #                               [[0, 1]],
        #                               [1.0],
        #                               [pairwise_feat],
        #                               intensity[r, c])
        #         factors_u.append(cur_f)
        #
        #
        #
        #
        # G = MarkovModel()
        # G.add_nodes_from(nodes)
        # print('Adding factors_u')
        # G.add_factors(*factors_u)
        # print('Adding factors_p')
        # G.add_factors(*factors_p)
        # print('Adding edges')
        # G.add_edges_from(edges_p)

        # checking if everthing is ok
        # print('Check model :', G.check_model())
        #
        # # initialize inference algorithm
        # denoise_infer = Mplp(G)
        #
        # # inferring MAP assignment
        # q = denoise_infer.map_query()

        cv.imshow('image', image)
        key = cv.waitKey(0)
        detection_prev = detection[0].Bbox
        img_prev = image
        hist_prev = hist


        if key == 27:
            break



if __name__ == '__main__':
    main()