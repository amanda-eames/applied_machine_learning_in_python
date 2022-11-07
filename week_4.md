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

Typically, Gaussian Naive Bayes is used for high-dimensional data. When each data instance has hundreds, thousands or maybe even more features. On the negative side, when the conditional independence assumption about features doesn't hold. In other words, for a given class, there's significant covariance among features, as is the case with many real world datasets. Other more sophisticated classification methods that can account for these dependencies are likely to outperform Naive Bayes.

#### Random Forests

Random Forests are an example of an ensemble. An ensemble takes multiple individual learning models and combines them to produce an aggregate model that is more powerful than any of its individual learning models alone. By combining different individual models into an ensemble, we can average out their individual mistakes to reduce the risk of overfitting while maintaining strong prediction performance. Recall that decision trees have a tendancy to overfit the training data, so the idea behind random forests is to have a collection of trees that do resonably well at prediction but are intentionally and randomly varied during the build. This variation happens in two ways, first the data selected to build each tree is random and second the feature chosen to split are selected randomly. 

```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier().fit(X_train, y_train)
print('Accuracy of RF classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of RF classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
```

| Pros | Cons|
|---|---|
|* Widely used and excellent prediction performance | * The resulting models are difficult to interpret|
|* Doesn't require careful normalisation of features or parametr tunning| * Like descion trees, not good choice for high dimentional tasks 
|* Easily parallelised across multiple CPU's ||
