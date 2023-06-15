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
    roi = image[y:y + h, x:x + w]

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
    # Wczytanie pliku
    with open("bboxes.txt", "r") as file:
        lines = file.readlines()
        line_index = 0

        # Zapis indeksów linii w pliku
        while line_index < len(lines):
            current_image_name = lines[line_index].strip()  #
            curr_image_dir = os.path.join(image_dir, current_image_name)

            # Pobranie danych z aktualnie przetwarzanego zdjęcia
            if str(curr_image_dir) == str(image_path):
                # Ilość Bboxów na zdjęciu
                num = int(lines[line_index + 1])
                Bboxes = []
                # Dla każdego Bboxa zapisuje współrzędne w liście
                for _ in range(num):
                    bbox_str = lines[line_index + 2]
                    bbox = list(map(float, bbox_str.strip().split()))
                    Bboxes.append(bbox)
                    line_index += 1

                # Zapisuje w liście Bboxy jako obiekty Detection
                detections.append(Detection(image_path, num, Bboxes))
                break
            else:
                line_index += 1

    # Zwraca listę obiektów
    return detections


# # Funkcja wczytująca jedynie 1 wartość z pliku Ground truth do porównania skuteczności algorytmu
# def load_gt(image_dir, image_path):
#     with open("bboxes_gt.txt", "r") as file:
#         lines = file.readlines()
#         line_index = 0
#         while line_index < len(lines):
#             current_image_name = lines[line_index].strip()
#             curr_image_dir = os.path.join(image_dir, current_image_name)
#             if str(curr_image_dir) == str(image_path):
#                 GT = []
#                 num = int(lines[line_index + 1])
#                 for i in range(num):
#                     value = lines[line_index + 2].split()
#                     GT.append(int(value[0]))
#                     line_index += 1
#             else:
#                 line_index += 1
#
#     # Zwraca wartości poprawnej klasyfikacji osób w liście
#     return GT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str)
    args = parser.parse_args()

    # Wczytanie ścieśki do zdjęć z podanego argumentu
    images_dir = Path(args.images_dir)
    images_paths = sorted([image_path for image_path in images_dir.iterdir() if image_path.name.endswith('.jpg')])

    # Sortowanie zdjęć alfabetycznie, inicjacja pustych list i zmiennych
    detection_prev = []
    hist = []
    first_frame = True
    all_bbox = 0
    correct = 0

    for image_path in images_paths:

        image = cv.imread(str(image_path))
        if image is None:
            # print(f'Error loading image {image_path}')
            continue

        # Dla każdego zdjęcia wczytuje Bboxy jako obiekty i dodaje je do listy Bboxów
        detection = load_bboxes(images_dir, image_path)
        hist_prev = hist.copy()     # Lista histogramów z poprzedniej klatki
        hist = []                   # Lista histogramów w obecnej klatce
        results = []
        # print(detection)
        iou = []                    # Lista wartości IoU
        center = []                 # Lista współrzędnych środka Bboxów w obecnej klatce
        center_prev = []            # Lista współrzędnych środka Bboxów w poprzedniej klatce
        distance = []               # Lista odległości pomiędzy Bboxami

        # Inicjacja pustej macierzy tranzycji
        matrix = - np.ones((len(detection[0].Bbox), len(detection_prev)))

        # W obecnej klatce obliczane są histogramy dla każdego Bboxa oraz wyznaczane są środki Bboxów
        for box in detection[0].Bbox:
            x, y, w, h = box[0], box[1], box[2], box[3]
            # print(x, y, w, h)
            center.append([x + 0.5 * w, y + 0.5 * h])
            cv.rectangle(image, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), (0, 255, 0), 2)

            # Histogramy z aktualnej klatki
            hist.append(calculate_histogram(image, box))

        matching_indices = []  # Indeksy dopasowanych elementów w macierzy

        # Wyznaczanie wartości dla pierwszej klatki
        if first_frame:
            detection_prev = detection[0].Bbox
            hist_prev.append(calculate_histogram(image, detection[0].Bbox[0]))
            img_prev = image
            first_frame = False
            matching_indices = [-1]

            # Wyświetlenie pierwszego wyniku w poprawnym formacie
            print(" ".join(map(str, matching_indices)) + "\n")

        # Śledzenie przechodniów dla pozostałych zdjęć
        else:
            for box_p in detection_prev:
                xx, yy, ww, hh = box_p[0], box_p[1], box_p[2], box_p[3]

                # IoU między 2 klatkami
                iou.append(bb_intersection_over_union(box_p, box))

                # Histogram poprzedniej klatki
                hist_prev.append(calculate_histogram(img_prev, box_p))
                center_prev.append([xx + 0.5 * ww, yy + 0.5 * hh])

            # TWORZENIE MACIERZY
            for i, now in enumerate(detection[0].Bbox):  # Wierszami są obecne Bboxy
                for j, prev in enumerate(detection_prev):  # Kolumnami sa poprzednie Bboxy
                    # print(now)

                    # Obliczenia wartości dla i*j Bboxów
                    if j < len(hist_prev) and i < len(hist):
                        # Wartość prawdopodobieństwa obliczona z porównania histogramów z klatki t-1 i t
                        result = cv.compareHist(hist_prev[j], hist[i], cv.HISTCMP_CORREL)

                        # Obliczanie dystansu między Bboxami w 2 kolejnych klatkach
                        distance.append(np.sqrt((center_prev[j][0] - center[i][0]) ** 2 + (center_prev[j][0] - center[i][1]) ** 2))
                        # print(distance)
                    else:
                        # Wypełnienie reszty macierzy wartościami -1
                        result = -1
                    # Zapis wartości przawdopodobieństwa w komórkach odpowiadających numerom wierszy i kolumn
                    matrix[i, j] = result

            # GREEDY SEARCH
            for i in range(len(matrix)):
                row = matrix[i]
                # Znajdowanie największej wartości w wierszu
                max_value = max(row)
                # Indeks największej wartości
                max_index = np.where(row == max_value)[0][0]
                # Jeśli dystans między Bboxami jest zbyt duży to jest to nowa osoba
                if max_value < 0.65 or distance[i] > 500:
                    max_index = -1

                # Wartości wyjściowe poszczególnych Bboxów
                matching_indices.append(max_index)


            # Metryka poprawnych dopasowań
            # gt = load_gt(images_dir, image_path)
            # all_bbox += len(gt)
            # for i in range(len(matching_indices)):
            #     # Porównanie dopasowania do wartości z pliku Ground Truth
            #     if gt[i] == matching_indices[i]:
            #         correct += 1

            # print("MATRIX")
            # print(matrix)
            # print(image_path)
            # print(f'GT: {gt}, ALL: {all_bbox}, CORRECT: {correct}')

            # Wyświetlenie końcowego wyniku w poprawnym formacie
            print(" ".join(map(str, matching_indices)) + "\n")
            # print(f'Correct percentage: {correct/all_bbox*100} %')


        # Wyświetlenie zdjęcia z narysowanymi Bboxami
        cv.imshow('image', image)

        # Podstawienie wartości z obecnej klatki jako wartości poprzedniej
        detection_prev = detection[0].Bbox
        img_prev = image
        hist_prev = hist

        # oczekiwanie na wciśnięcie klawisza (w trakcie testów)
        # key = cv.waitKey(0)
        # if key == 27:
        #     break


if __name__ == '__main__':
    main()
