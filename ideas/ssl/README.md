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