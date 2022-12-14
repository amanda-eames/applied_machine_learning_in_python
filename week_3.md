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

Recall or True Positive Rate (TPR) is the fraction of all positive instances the classifer correctly identifies as positive. We can increase the recall by increasing the number of TP or reducing the number of FN

$$ Recall = TP/(TP+FN) $$

Another metrics is the False Positive Rate (FPR) fraction of all negative instances does the classifier incorrectly identify as positive 

$$ FPR = FP/(TN+FP)$$

There is a tradeoff between recall and precison and need to balance what is more important for the application. We have another metric that is the harmonic mean of precision and recall, called F1 score

$$F_1 = 2 \dfrac{Precision \cdot Recall}{Precision + Recall} = \dfrac{2TP}{2TP+FN+FP}$$

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

accuracy_score(y_test, y_predicted)
classification_report(y_test, y_predicted, target_names=['not 1', '1'])
```

### Precision-recall and ROC curves

```python
y_scores_m = m.fit(X_train, y_train).decision_function(X_test)
precision, recall, thresholds = precision_recall_curve(y_test, y_scores_m)
plt.plot(precision, recall)
plt.xlabel('precision')
plt.ylabel('recall')
plt.show()
```

### Multi-Class Evaluation
These are just an extention of the binary case, overall evaluation metrics are averaged across all classes and there are different ways to do this. 

### Regression Evaluation
Typically the r2_score is typically enough, and can be negative. The other alternatives are mean_absolute_error, mean_squared_error and median_absolute error (robust to outliers)

```python
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score

lm_dummy_mean = DummyRegressor(strategy = 'mean').fit(X_train, y_train)
y_predict_dummy_mean = lm_dummy_mean.predict(X_test)
mean_squared_error(y_test, y_predict_dummy_mean)
r2_score(y_test, y_predict_dummy_mean)
```
