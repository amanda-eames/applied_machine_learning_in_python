## Introduction

# Terms:

* Feature Representation: a set of techniques that allows a system to automatically discover the representations needed for feature detection or classification from raw data
* Lable: the target value, in classification it is the lable for of an object and regression it is the continuous value
* Training and test set: usually a 75 to 25 split. The training set is used to estimate the parameters of the model, the test set is used to evaulate the model on unseen data. 
* Overfitting: when the increase in the model complexity decreases the performance of the model, no longer captures the global patterns. 
* Underfitting: when the model accuracy would benifit from more complexity. 

# Models:

* Classification: the target value is a discrete class value, and in binary classification the target value is 0 (negative class) or 1 (positive class). There is also, multi-class classification where the target value can be a set of discrete values. The third, is multi-label classification where the target value can have multiple assignments. 
* Regression: the target value is a continuous value. 

# Predicition Methods:

# k-Nearest Neighbour

Given a training set x_train and labels y_train, to classify a new instance x_test:

* Find the most similar instances (say X_NN) to the x_test that are in x_train
* Get the labels y_NN for the instances in X_NN
* Predict the label for x_test by combining the labels y_NN
* As we increase k, single training data points no longer have as dramatic an influence on the prediction. The result is a much smoother decision boundary, which represents a model with lower model complexity 
* Can be used for regression as well, this looks like steps along the regression line. 
* The R^2 regression score or the coefficient of determination, measures how well a predicted model for regression fits the data. The score is between 0 and 1, where 1 mean the model fits the data perfectly.

