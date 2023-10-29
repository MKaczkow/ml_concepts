### Useful links:
* https://datascience.stackexchange.com/questions/31041/what-does-logits-in-machine-learning-mean
* https://www.quora.com/What-are-Logits-in-deep-learning
* 

### Po ludzku
Po ludzku, to **logit** to funkcja, która mapuje *[0, 1]* na *[-inf, inf]*. Czyli odwrotnie niż  sigmoid. W ML jest to wektor nieznormalizowanych predykcji, które generuje klasyfikator. Normalnie używa się zlogarytmowanej wersji, ponieważ jest bardziej numerycznie stabilna i różniczkowalna.  

Albo inaczej, to **logit** w ML to wyjście z ostatniej warstwy sieci neuronowej, przed zastosowaniem funkcji aktywacji, czyli znowu, nieznormalizowane.
