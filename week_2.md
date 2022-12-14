## Terms:

* Feature Representation: a set of techniques that allows a system to automatically discover the representations needed for feature detection or classification from raw data
* Lable: the target value, in classification it is the lable for of an object and regression it is the continuous value
* Training and test set: usually a 75 to 25 split. The training set is used to estimate the parameters of the model, the test set is used to evaulate the model on unseen data. 
* Overfitting: when the increase in the model complexity decreases the performance of the model, no longer captures the global patterns. 
* Underfitting: when the model accuracy would benifit from more complexity. 

## Models:

* Classification: the target value is a discrete class value, and in binary classification the target value is 0 (negative class) or 1 (positive class). There is also, multi-class classification where the target value can be a set of discrete values. The third, is multi-label classification where the target value can have multiple assignments. 
* Regression: the target value is a continuous value. 

## k-Nearest Neighbour

Given a training set x_train and labels y_train, to classify a new instance x_test:

* Find the most similar instances (say X_NN) to the x_test that are in x_train
* Get the labels y_NN for the instances in X_NN
* Predict the label for x_test by combining the labels y_NN
* As we increase k, single training data points no longer have as dramatic an influence on the prediction. The result is a much smoother decision boundary, which represents a model with lower model complexity 
* Can be used for regression as well, this looks like steps along the regression line. 
* The R^2 regression score or the coefficient of determination, measures how well a predicted model for regression fits the data. The score is between 0 and 1, where 1 mean the model fits the data perfectly.
* Pro's: simple and easy to understand why a prediction was made, and can be a good baseline to compare more sophisticaed methods. 
* Con's when training data has alot of instances, or instance has alot of features this can impact heavily on performace (particularily if your data is sparse)

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
knnreg = KNeighborsRegressor(n_neighbors = 5).fit(X_train, y_train)
knnreg.predict(X_test)
knnreg.score(X_test, y_test)
```

## Linear Regression: Least-Squares
* A linear model is a sum of wieghted variables that predict a target output value given the input data instance. Least-squares, minimises the sum of squared differences between the predicted target values and actual values.
* The learning algorithm finds the parameters tht optimise an objective function, typically to minimise some kind of loss function of the predicted target values vs actual target values (i.e., some penalty function) 

```python
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 0)
linreg = LinearRegression().fit(X_train, y_train)
linreg.coef_
linreg.intercept_
linreg.score(X_train, y_train)
linreg.score(X_test, y_test)
```


## Ridge Regression
* Uses the same least-squares criterion but adds a pentaly for larger variations in weights, the addition of a penalty paramter is called regulisation. This helps to reduce overfitting by resticting the models complexity. 
* Uses L2 regulisation, sum squares in weights. The influence of regularisation term is controled by a coefficient alpha. Higher alpha means more regulisation which means a simplier model
* Rational: large weights means the sum of there squared valued is large
* Given the features can have different scales, we need to normalise the data so that ridge regression can behave more fairly for each feature. An exmaple of normaliasation is call min max scalling. 

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime,random_state = 0)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linridge = Ridge(alpha=20.0).fit(X_train_scaled, y_train)
linridge.intercept_
linridge.coef_
linridge.score(X_train_scaled, y_train)
```

## Lasso Regression
* Is an L1 penalty, take the absolute value of the weights rather then squared. The effect has the effect of setting parameter weight to zero, called a sparse solution a kind of feature selection
* When we have many small/medium sized effects use ridge and when few variables with medium large effects use lasso.

```python
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 0)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linlasso = Lasso(alpha=2.0, max_iter = 10000).fit(X_train_scaled, y_train)
```

## Non-linear realtionships
* Polynomial features, we can apply non-linear transformations to create a new features that capture more complex relationships. Note need to keep in mind of polynomial feature expansion, this can lead to overfit. 
* Logistic Regression is used for classification, apply a log function to compress the target values to between 0 and 1, naturally this be interpreted as a proabability. For binary classification, we interpret the value as the porbability of belonging to the positive class. We can also apply a the same penalty as L2 regularisation (this is turned on by default in sklearn)

```python
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_F1)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y,random_state = 0)
linreg = LinearRegression().fit(X_train, y_train)

```

## Support Vector Machines
* Linear classification used a feature vector and the target value class is determined by the sign. Note this is dot product. 
$$f(x,w,b) = sign(w \cdot x + b)$$
* To evaluate the classifier one meteric is classifier margin, as the maximum width the decision boundary area can be increased before hitting a data point. The linear classifier with maximum margin is a linear support vector machine (LSVM)
* Regualisation strength is determined by the C parameter, larger values mean less regularisation fits the training data well and smaller values are more tolerant of errors.  
* Pro's: simple and easy to train, fast predictions, scales well to large datasets, works well with sparse data, can interpret results.
* Con's for lower-dimentional data other methods are more suitable, data may not be linearly seperable. 
* This approach can be used for multi-class classification, have a binary classifier for each class, either belong to the class or not. Then we get the training data, and passed to each of the class classifiers and then take the one with the highest probability.

```python
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
clf = SVC(kernel = 'linear', C=1).fit(X_train, y_train)
clf.score(X_train, y_train)
clf.score(X_test, y_test)
```

```python
# multi-class classification
from sklearn.svm import LinearSVC
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
clf = LinearSVC(C=5, random_state = 67).fit(X_train, y_train)
print('Coefficients:\n', clf.coef_)
print('Intercepts:\n', clf.intercept_
```

## Kernalised Support Vector Machines
* This is used when linear support vector machines are not complex enough to capture the boundary. In essence, one way to think about what kernelized SVMs, is they take the original input data space and transform it to a new higher dimensional feature space, where it becomes much easier to classify the transform to data using a linear classifier.
* Radial basis function kernal: what this looks like is all the point inside a certain radius are map to the same area in the feature space.  
* The kernel trick, is that internally, the algorithm doesn't have to perform this actual transformation on the data points to the new high dimensional feature space. Instead, the kernelized SVM can compute these more complex decision boundaries just in terms of similarity calculations between pairs of points in the high dimensional space where the transformed feature representation is implicit.
* Can also have polynomial SVM, here the tranformation will be a polnomial of certain degree for the kernel.
* The Gamma parameter is the kernel width parameter, which affects how tightly the decision boundaries end up surrounding points in the input space. Small gamma means a larger similarity radius. So that points farther apart are considered similar. Which results in more points being group together and smoother decision boundaries.
* Pro's: Perform well on a range of daatsets, versatile different kernal function, work well for both low and high dimensional data
* Con's: Efficiency decreases as tranining set size increases, need careful nomalisation and parameter tuning, does not provide direct probabilty estimates, difficult to interpet why a prediction was made. 

$$ K(x,x^{\prime}) = exp \[- \gamma \cdot ||x - x^{\prime} || ^2 \]$$

```python
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
SVC(kernel='rbf', C=1), X_train3, y_train3)
SVC(kernel = 'poly', degree = 3).fit(X_train, y_train)
```

## Cross-Validation
*  Cross-validation is a method that goes beyond evaluating a single model using a single Train/Test split of the data by using multiple Train/Test splits, each of which is used to train and evaluate a separate model
*   Cross-validation basically gives more stable and reliable estimates of how the classifiers likely to perform on average by running multiple different training test splits and then averaging the results, instead of relying entirely on a single particular training set.
*   The most common type of cross-validation is k-fold cross-validation most commonly with K set to 5 or 10. For example, to do five-fold cross-validation, the original dataset is partitioned into five parts of equal or close to equal size.
*   This extra information does come with extra cost. It does take more time and computation to do cross-validation.
*   The Stratified Cross-validation means that when splitting the data, the proportions of classes in each fold are made as close as possible to the actual proportions of the classes in the overall data set as shown here

```python
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve

param_range = np.logspace(-3, 3, 4)
train_scores, test_scores = validation_curve(SVC(), X, y,
                                            param_name='gamma',
                                            param_range=param_range, cv=3)
```

## Decision Trees
*  The generalisation of finding a set of rules that can learn to categorize an object into the correct category to many other classification tasks. 
*  Pure node (all one class, perfect classification) mixed node (mixture of classes), teh prediction will just be the majority class at that node. 
*  Can also be used for regression. 
*  Typically such trees are overly complex and essentially memorized the training data. So when building decision trees, we need to use some additional strategy to prevent this overfitting
*  One strategy to prevent overfitting is to prevent the tree from becoming really detailed and complex by stopping its growth early. This is called pre-pruning. Another strategy is to build a complete tree with pure leaves but then to prune back the tree into a simpler form. 

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X.data, y.target, random_state = 3)
clf = DecisionTreeClassifier(max_depth = 3).fit(X_train, y_train)
clf.score(X_train, y_train)
clf.score(X_test, y_test)
feat_importance = clf.tree_.compute_feature_importances(normalize=True)
```

