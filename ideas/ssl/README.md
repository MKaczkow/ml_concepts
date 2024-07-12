# Self-supervised learning

* example library: [lightly](https://docs.lightly.ai/self-supervised-learning/index.html)
* run interface:
    - `cd ideas/ssl`
    - `mlflow ui`

# TODO
- [x] logować też loss i więcej metryk do mlflow
    - [x] loss
    - co jeszcze? (based on: [geeksforgeeks](https://www.geeksforgeeks.org/clustering-performance-evaluation-in-scikit-learn/))
        - [x] ARI  (??)
        - [x] Davies-Bouldin index (??)
        - [x] Silhouette score (??)
- [x] czytanie
    - [x] Understanding Contrastive Representation Learning through Alignment and Uniformity
    - [x] intriguig properties of contrastive loss
    - [x] 'perfectly balanced'
    - [x] NXTLoss (kinda)
- [x] sprawdzić mniejsze embeddingi
    - [x] jak się zachowuje loss w zależności od embedding size?
    - [x] re-run wszystkich eksperymentów z zapisem loss
- [x] Jakie są rozsądne wartości lossa dla tego zadania? (wprost w papierze nie jest podane, ocena tylko na downstream tasks, trzeba porównać 'wewnątrz' eksperymentu, np. pomiędzy różnymi wymiarowościami latent space)
- [x] założyć dokument google, który będzie pełnił funkcję "dziennika" z eksperymentów, takie podkreślenie key point'ów z eksperymentów
- [x] poprawić literówkę w nazwie notebooka (jak się skończy run)
- [x] update

## Papiery
* Czy są jakieś augementacje do danych typu 'seria', np. zamiana kolejności kilku elementów w sekwencji i/lub różny szum do kolejnych elementów ?
    - https://arxiv.org/abs/2304.14601
    - https://arxiv.org/abs/2211.04888
    - https://arxiv.org/abs/2206.15015
    - https://arxiv.org/pdf/2012.03457
* 'Seria' o podstawach `CL` (contrastive learning)
    - https://arxiv.org/pdf/2005.10242
    - https://arxiv.org/pdf/2011.02803

## Hiperparametry
- embedding_space_size [default(128), 32, 16, 8]
- num_classes [default(3), 5, 10]
- n_components_PCA [default(50), 20, 7, 3]

## Loss etc.
* [gh issue on entropy](https://github.com/google-research/simclr/issues/188)
* [gh issue on distillation](https://github.com/google-research/simclr/issues/159)

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
