import numpy as np
import os
from collections import namedtuple
import cv2 as cv
import argparse
from pathlib import Path

Detection = namedtuple("Detection", ["image_path", "num", "Bbox"])


# metoda działa jedynie dla Bboxów blisko siebie
def bb_intersection_over_union(bbox1, bbox2):
    # Określa współrzędne Bboxów z danych
    x1_min, y1_min, x1_max, y1_max = bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]
    x2_min, y2_min, x2_max, y2_max = bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]
    # Współrzędne Bboxów  między sobą
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    # Oblicza maksymalną wartość pola
    intersection_area = max(0, x_inter_max - x_inter_min) * max(0, y_inter_max - y_inter_min)

    # Oblicz pola powierzchni obu bounding boxów
    bbox1_area = bbox1[2] * bbox1[3]
    bbox2_area = bbox2[2] * bbox2[3]

    # Oblicz IoU
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    return iou


# Funkcja oblicza histogram z jednego Bboxa
def calculate_histogram(image, bbox):
    # Zapisuje obszar Bboxa
    x, y, w, h = map(int, bbox)
    roi = image[y:y+h, x:x+w]

    # Konwertuje obszar na przestrzeń barw HSV
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

    # Oblicza histogram kolorów w przestrzeni HSV
    hist = cv.calcHist([hsv_roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    # Normalizuje histogram i zwraca wartość
    cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)

    return hist.flatten()


# Funkcja wczytująca dane Bounding boxów z pliku .txt
def load_bboxes(image_dir, image_path):
    detections = []
    with open("bboxes.txt", "r") as file:                       # Wczytanie pliku
        lines = file.readlines()
        line_index = 0
        while line_index < len(lines):                           # Zapis indeksów linii w pliku
            current_image_name = lines[line_index].strip()       #
            curr_image_dir = os.path.join(image_dir, current_image_name)
            if str(curr_image_dir) == str(image_path):           # Pobranie danych z aktualnie przetwarzanego zdjęcia
                num = int(lines[line_index + 1])                 # Ilość Bboxów na zdjęciu
                Bboxes = []
                for _ in range(num):                             # Dla każdego Bboxa zapisuje współrzędne w liście
                    bbox_str = lines[line_index + 2]
                    bbox = list(map(float, bbox_str.strip().split()))
                    Bboxes.append(bbox)
                    line_index += 1
                detections.append(Detection(image_path, num, Bboxes))   # Zapisuje w liście Bboxy jako obiekty Detection
                break
            else:
                line_index += 1
    return detections  # Zwraca listę obiektów


# Funkcja wczytująca jedynie 1 wartość z pliku Ground truth do porównania skuteczności algorytmu
def load_gt(image_dir, image_path):
    with open("bboxes_gt.txt", "r") as file:
        lines = file.readlines()
        line_index = 0
        while line_index < len(lines):
            current_image_name = lines[line_index].strip()
            curr_image_dir = os.path.join(image_dir, current_image_name)
            if str(curr_image_dir) == str(image_path):
                GT = []
                num = int(lines[line_index + 1])
                for i in range(num):
                    value = lines[line_index+2].split()
                    GT.append(int(value[0]))
                    line_index += 1
            else:
                line_index += 1
    return GT                       # Zwraca wartości poprawnej klasyfikacji osób w liście


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str)
    args = parser.parse_args()
    images_dir = Path(args.images_dir)               # Wczytanie ścieśki do zdjęć z podanego argumentu
    images_paths = sorted([image_path for image_path in images_dir.iterdir() if image_path.name.endswith('.jpg')])
    detection_prev = []     # Sortowanie zdjęć alfabetycznie, inicjacja pustych list i zmiennych
    hist = []
    first_frame = True
    all_bbox = 0
    correct = 0

    for image_path in images_paths:

        image = cv.imread(str(image_path))
        if image is None:
            # print(f'Error loading image {image_path}')
            continue

        detection = load_bboxes(images_dir,image_path)      # Dla każdego zdjęcia wczytuje Bboxy jako obiekty i dodaje je do listy Bboxów
        hist_prev = hist.copy()     # Lista histogramów z poprzedniej klatki
        hist = []   # Lista histogramów w obecnej klatce
        results = []
        # print(detection)
        iou = []    # Lista wartości IoU
        center = []          # Lista współrzędnych środka Bboxów w obecnej klatce
        center_prev = []     # Lista współrzędnych środka Bboxów w poprzedniej klatce
        distance = []        # Lista odległości pomiędzy Bboxami
        matrix = - np.ones((len(detection[0].Bbox), len(detection_prev)))   # Inicjacja pustej macierzy tranzycji
        for box in detection[0].Bbox:                       # W obecnej klatce obliczane są histogramy dla każdego Bboxa
            x, y, w, h = box[0], box[1], box[2], box[3]     # oraz wyznaczane są środki Bboxów
            # print(x, y, w, h)
            center.append([x+0.5*w, y+0.5*h])
            cv.rectangle(image, (int(x), int(y)), (int(x)+int(w), int(y)+int(h)), (0, 255, 0), 2)
            hist.append(calculate_histogram(image, box))            # Histogramy z aktualnej klatki

        matching_indices = []  # Indeksy dopasowanych elementów w macierzy

        if first_frame:        # Wyznaczanie wartości dla pierwszej klatki
            detection_prev = detection[0].Bbox
            hist_prev.append(calculate_histogram(image, detection[0].Bbox[0]))
            img_prev = image
            first_frame = False
            matching_indices = [-1]
            print(" ".join(map(str, matching_indices)) + "\n")  # Wyświetlenie pierwszego wyniku w poprawnym formacie

        else:   # Śledzenie przechodniów dla pozostałych zdjęć

            for box_p in detection_prev:
                xx, yy, ww, hh = box_p[0], box_p[1], box_p[2], box_p[3]
                iou.append(bb_intersection_over_union(box_p, box))      # IoU między 2 klatkami
                hist_prev.append(calculate_histogram(img_prev, box_p))  # Histogram poprzedniej klatki
                center_prev.append([xx+0.5*ww, yy+0.5*hh])


            # TWORZENIE MACIERZY
            for i, now in enumerate(detection[0].Bbox):     # Wierszami są obecne Bboxy
                for j, prev in enumerate(detection_prev):   # Kolumnami sa poprzednie Bboxy
                    # print(now)
                    if j < len(hist_prev) and i < len(hist):   # Obliczenia wartości dla i*j Bboxów
                        result = cv.compareHist(hist_prev[j], hist[i], cv.HISTCMP_CORREL)   # Wartość prawdopodobieństwa obliczona z porównania histogramów z klatki t-1 i t
                        distance.append(np.sqrt((center_prev[j][0]-center[i][0])**2 + (center_prev[j][0]-center[i][1])**2)) # Obliczanie dystansu między Bboxami w 2 kolejnych klatkach
                        # print(distance)

                    else:
                        result = -1         # Wypełnienie reszty macierzy wartościami -1
                    matrix[i, j] = result    # Zapis wartości przawdopodobieństwa w komórkach odpowiadających numerom wierszy i kolumn

            # GREEDY SEARCH
            for i in range(len(matrix)):
                row = matrix[i]
                max_value = max(row)  # Znajdowanie największej wartości w wierszu
                max_index = np.where(row == max_value)[0][0]  # Indeks największej wartości
                if max_value < 0.65 or distance[i] > 500:     # Jeśli dystans między Bboxami jest zbyt duży to jest to nowa osoba
                    max_index = -1
                matching_indices.append(max_index)              # Wartości wyjściowe poszczególnych Bboxów

            gt = load_gt(images_dir, image_path)                # Metryka poprawnych dopasowań
            all_bbox += len(gt)
            for i in range(len(matching_indices)):
                # print(f'gti{gt[i]}, matching_i {matching_indices[i]}')
                if gt[i] == matching_indices[i]:               # Porównanie dopasowania do wartości z pliku Ground Truth
                    correct += 1


            # print("MATRIX")
            # print(matrix)
            # print(image_path)
            # print(f'GT: {gt}, ALL: {all_bbox}, CORRECT: {correct}')
            print(" ".join(map(str, matching_indices))+"\n")        # Wyświetlenie końcowego wyniku w poprawnym formacie
            # print(f'Correct percentage: {correct/all_bbox*100} %')

        cv.imshow('image', image)           # Wyświetlenie zdjęcia z narysowanymi Bboxami
        detection_prev = detection[0].Bbox  # Podstawienie wartości z obecnej klatki jako wartości poprzedniej
        img_prev = image
        hist_prev = hist

        # key = cv.waitKey(0)   # oczekiwanie na wciśnięcie klawisza (w trakcie testów)
        # if key == 27:
        #     break



if __name__ == '__main__':
    main()