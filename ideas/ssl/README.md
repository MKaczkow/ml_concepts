# Self-supervised learning

* example library: [lightly](https://docs.lightly.ai/self-supervised-learning/index.html)

# TODO:
batch danych 256 z 3-4 klasami
* czy spada loss 
* czy rosną dystanse między klasami
- odległości
- loss
- wizualizacje
    - tuż po inicjalizacji
    - wandb

1. Przygotowanie zbioru danych, tak żeby każdy obrazek miał w nazwię klasę i jakiś identyfikator, np `0_123.jpg`
2. Trening modelu na tym zbiorze (`LightlyDataset`)
3. Generacja embeddingów dla całego zbioru (może być od razu z zapisem albo funkcja, ale wtedy musi zwracać także filename)
4. Wizualizacja wyników (PCA / t-SNE + K-means), gdzie label bierzemy z nazwy pliku

# TODO:
dla większej ilości klas: np. 3, 5, 10
- wyświetlanie pierwsze 2 kierunki pca (+ mozę explained_variance_ratio)
- wyświetlanie 2 kierunki tsne

- sprawdzić czy n_components w PCA wpływa na wyjaśnianie przez pierwsze kilka kierunków (np. 2-3)

- mniejszy embedding space, np. 32, 16, 8

- czy są jakieś augementacje do danych typu 'seria', np. zamiana kolejności kilku elementów w sekwencji i/lub różny szum do kolejnych elementów 

- jakiś papier do powyższego?? 
    - https://arxiv.org/abs/2304.14601
    - https://arxiv.org/abs/2211.04888
    - https://arxiv.org/abs/2206.15015

## Hiperparametry
- embedding_space_size [default(128), 32, 16, 8]
- num_classes [default(3), 5, 10]
- n_components_PCA [default(50), 20, 7, 3]

## Do mlflow
- parametry:
    - wielkość embedding space
    - liczba klas
    - n_components PCA
- artefakty:
    - wizualizacja PCA (pierwsze 2 kierunki)
    - wizualizacja t-SNE (pierwsze 2 kierunki)
- metryki
    - explained_variance_ratio