# Introduction

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

## Linear Regression: Least-Squares
* A linear model is a sum of wieghted variables that predict a target output value given the input data instance. Least-squares, minimises the sum of squared differences between the predicted target values and actual values.
* The learning algorithm finds the parameters tht optimise an objective function, typically to minimise some kind of loss function of the predicted target values vs actual target values (i.e., some penalty function) 


## Ridge Regression
* Uses the same least-squares criterion but adds a pentaly for larger variations in weights, the addition of a penalty paramter is called regulisation. This helps to reduce overfitting by resticting the models complexity. 
* Uses L2 regulisation, sum squares in weights. The influence of regularisation term is controled by a coefficient alpha. Higher alpha means more regulisation which means a simplier model
* Rational: large weights means the sum of there squared valued is large
* Given the features can have different scales, we need to normalise the data so that ridge regression can behave more fairly for each feature. An exmaple of normaliasation is call min max scalling. 

## Lasso Regression
* Is an L1 penalty, take the absolute value of the weights rather then squared. The effect has the effect of setting parameter weight to zero, called a sparse solution a kind of feature selection
* When we have many small/medium sized effects use ridge and when few variables with medium large effects use lasso.

## Non-linear realtionships
* Polynomial features, we can apply non-linear transformations to create a new features that capture more complex relationships. Note need to keep in mind of polynomial feature expansion, this can lead to overfit. 
* Logistic Regression is used for classification, apply a log function to compress the target values to between 0 and 1, naturally this be interpreted as a proabability. For binary classification, we interpret the value as the porbability of belonging to the positive class. We can also apply a the same penalty as L2 regularisation (this is turned on by default in sklearn)

## Support Vector Machines
* Linear classification used a feature vector and the target value class is determined by the sign. Note this is dot product. 
$$f(x,w,b) = sign(w \cdot x + b)$$
* To evaluate the classifier one meteric is classifier margin, as the maximum width the decision boundary area can be increased before hitting a data point. The linear classifier with maximum margin is a linear support vector machine (LSVM)
* Regualisation strength is determined by the C parameter, larger values mean less regularisation fits the training data well and smaller values are more tolerant of errors.  
* Pro's: simple and easy to train, fast predictions, scales well to large datasets, works well with sparse data, can interpret results.
* Con's for lower-dimentional data other methods are more suitable, data may not be linearly seperable. 
* This approach can be used for multi-class classification, have a binary classifier for each class, either belong to the class or not. Then we get the training data, and passed to each of the class classifiers and then take the one with the highest probability.

## Kernalised Support Vector Machines
* This is used when linear support vector machines are not complex enough to capture the boundary. In essence, one way to think about what kernelized SVMs, is they take the original input data space and transform it to a new higher dimensional feature space, where it becomes much easier to classify the transform to data using a linear classifier.
* Radial basis function kernal: what this looks like is all the point inside a certain radius are map to the same area in the feature space.  
* The kernel trick, is that internally, the algorithm doesn't have to perform this actual transformation on the data points to the new high dimensional feature space. Instead, the kernelized SVM can compute these more complex decision boundaries just in terms of similarity calculations between pairs of points in the high dimensional space where the transformed feature representation is implicit.
* Can also have polynomial SVM, here the tranformation will be a polnomial of certain degree for the kernel.
* The Gamma parameter is the kernel width parameter, which affects how tightly the decision boundaries end up surrounding points in the input space. Small gamma means a larger similarity radius. So that points farther apart are considered similar. Which results in more points being group together and smoother decision boundaries.

$$ K(x,x^{\prime}) = exp \[- \gamma \cdot ||x - x^{\prime} || ^2 \]$$
