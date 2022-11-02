# Model Evaluation & Selection

To understand an application overall performance need to it's ability to meet certain goals, while model accuracy is important often need to score card of metrics to 
evaulate performance e.g., user satisfaction, patient survival rate etc. For example, in a health application that uses a classifier to detect tumors in a medical image, 
we may want to flag anything that even has a small chance of being cancerous. Even if it means sometimes incorrectly classifying healthy tissue as diseased.

### Imbalanced Classes

By way of example, imagine we have a classifier for fraudulent actitivity we would expect 99% of transaction classified in the positive class (regular authorised 
activity). However, the other 1% may be fruadulent, if we evalute performance on accuracy alone even a classifier that just assigns everything to the postitive class 
will have an accuracy of 99%.

In a binary classifier we have four scenarios, TP (True Positive), TN (True Negative), FP (False Positive, Type 1 error) and FN (False Negative, Type II error)

```python
from sklearn.dummy import DummyClassifier

dummy_majority = DummyClassifier(strategy = 'most_frequent' # predicts the most frequent label in training set
                                #strategy = 'stratified' random predictions based on training set class distribution
                                #strategy = 'uniform' uniformly random predictions
                                #strategy = 'constant' predicts based on a constant label provided
                                ).fit(X_train, y_train)
dummy_majority.score(X_test, y_test)
```

