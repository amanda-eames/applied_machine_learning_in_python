# Model Evaluation & Selection

To understand an application overall performance need to it's ability to meet certain goals, while model accuracy is important often need to score card of metrics to 
evaulate performance e.g., user satisfaction, patient survival rate etc. For example, in a health application that uses a classifier to detect tumors in a medical image, 
we may want to flag anything that even has a small chance of being cancerous. Even if it means sometimes incorrectly classifying healthy tissue as diseased.

### Imbalanced Classes

By way of example, imagine we have a classifier for fraudulent actitivity we would expect 99% of transaction classified in the positive class (regular authorised 
activity). However, the other 1% may be fruadulent, if we evalute performance on accuracy alone even a classifier that just assigns everything to the postitive class 
will have an accuracy of 99%.

In a binary classifier we have four scenarios, TP (True Positive), TN (True Negative), FP (False Positive, Type 1 error) and FN (False Negative, Type II error). This can be extended to k-classifier be rather you would have a k by k matrix. The count for each outcome is called a Confusion Matrix

```python
from sklearn.dummy import DummyClassifier

dummy_majority = DummyClassifier(strategy = 'most_frequent' # predicts the most frequent label in training set
                                #strategy = 'stratified' random predictions based on training set class distribution
                                #strategy = 'uniform' uniformly random predictions
                                #strategy = 'constant' predicts based on a constant label provided
                                ).fit(X_train, y_train)
dummy_majority.score(X_test, y_test)
```

```python
from sklearn.tree import DecisionTreeClassifier

model = 
md = model.fit(X_train, y_train)
model_predicted = md.predict(X_test)
confusion = confusion_matrix(y_test, model_predicted)
```

### Confusion Matrices & Basic Evaluation Metrics

Recall or True Positive Rate (TPR) is the fraction of all positive instances the classifer correctly identifies as positive 

$$ Recall = TP/TP+FN $$
