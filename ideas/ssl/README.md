# Self-supervised learning

* example library: [lightly](https://docs.lightly.ai/self-supervised-learning/index.html)
* run interface:
    - `cd ideas/ssl`
    - `mlflow ui`

# TODO
* założyć dokument google, który będzie pełnił funkcję "dziennika" z eksperymentów, takie podkreślenie key point'ów z eksperymentów
* logować też loss i więcej metryk do mlflow
* jakie są rozsądne wartości lossa dla tego zadania
* sprawdzić mniejsze embeddingi
    - jak się zachowuje loss w zależności od embedding size?
* poprawić literówkę w nazwie notebooka (jak się skończy run)

## Papiery
* Czy są jakieś augementacje do danych typu 'seria', np. zamiana kolejności kilku elementów w sekwencji i/lub różny szum do kolejnych elementów ?
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
