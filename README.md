Program ma za zadanie śledzić przechodniów na kolejnych klatkach wykorzystując graf dwudzielny (Bipartite graph).

Zasada działania algorytmu:
Wczytywane są zdjęcia po kolei z podanego folderu, na których znajdują się prostokąty ograniczające przechodniów. 
Współrzędne Bboxów z aktualnego zdjęcia wczytywane są z pliku .txt.
Następnie w każdym prostokącie obliczany jest jego histogram.
Prawdopodobieństwo tego, że dany przechodzień z poprzedniej klatki jest tym wyznaczonym w obecnej klatce obliczane jest za pomocą porównania 2 histogramów.
Prawdopodobieństwo Bboxów liczone jest metodą "każdy z każdym". 
Tworzona jest macierz tranzycji, której wiersze określają Bboxy obecnej klatki, a kolumny Bboxy z poprzedniej klatki.
W macierzy wypełniane są komórki odpowiadające numerom kolejnych Bboxów.
Dopasowanie Bounding Boxów między kolejnymi klatkami wybierane jest metodą Greedy Search.

Sama metoda porównania histogramów jest niewystarczająco skuteczna, ponieważ opiera się wyłącznie na porównaniu kolorów pikseli.
W przypadku identyfikacji osób, histogramy mogą być bardzo podobne do siebie, dlatego że 2 różne osoby mogą być podobnej wielkości na zdjęciu oraz mają pdobny kształ.
Jeżeli na zdjęciu znajduje się więcej osób lub osoby są podobnie ubrane (ten sam kolor koszulki/spodni) histogramy zwracają bardzo duże prawdopodobieństwo nawet dla 2 różnych osób.
Algorytm złożony wyłącznei z porównania histogramów na bazie testowej osiągnął 58%  skuteczności dopasowania.

W celu poprawy skuteczności w programie zostało dodane sprawdzanie odległości pomiędzy środkami Bounding boxów z klatki obecnej i poprzedniej.
Odległość została obliczona metodą Euklidesową. Jeżeli odległość między środkami Bboxów jest większa niż 500 to algorytm uznaje Bbox z obecnej klatki jako nowa osoba, która nie pojawiła się wcześniej na obrazie.

Po zastosowaniu dodatkowego parametru przy kalsyfikacji osób skutecznośc identyfikacji wzrosła na zbiore testowym do 80% dokładności. 

Dalszy rozwój algorytmu polegałby na dodaniu kolejnych cech obrazowych, tak aby obliczanie prawdopodobieństwa było dokładniejsze.
Znaczącą poprawę mogłoby dać wyznaczenie ilości detektorów np: SIFT w klatce obecnej i poprzedniej oraz porównanie ilości dobrze dopasowanych deskryptorów pomiędzy nimi.
Im większa liczba poprawnych dopasowań, tym większe prawdopodobieństwo identyfikacji osoby.

Na zbiorze testowym sprawdzono także porównanie Bboxów metodą IoU (Intersection over Union), jednak nie dała ona pozytywnych rezultatów.
Metoda ta działa wyłacznie, jeśli klatki są w równych, krótkich odstępach czasowych po sobie, a osoby w kolejnych klatkach nie przemieszczają sie za daleko. 
Jeżeli osoba na dwóch kolejnych klatkach przemieszcza się dalej niż obejmuje to Bounding Box, to porównanie IoU daje wartość 0. 

Zaproponowana metoda porównania histogramów oraz odległości między środkami Bboxów daje zadowalające rezultaty na poziomie 80% dokładności. 