This module covers more advanced supervised learning methods that include ensembles of trees (random forests, gradient boosted trees), and neural networks.
.
#### Naive Bayes Classifiers

Naive Bayes classifiers are called naive because they make the assumption that each feature of an instance is independent of all the others, given the class.
This is not always the case with features, in reality there can be correlations but with this assumption we get highly efficient learning and prediction but the 
generalisation performance may be worse then  more sophisticated learning models. 

Types:
* Bernoulli: binary features e.g., work presence 
* Multinomial: discrete features e.g., word count
* Guassian continuous/real valued features e.g., for each feature mean and standard deviation

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
nbclf = GaussianNB().fit(X_train, y_train)
```
