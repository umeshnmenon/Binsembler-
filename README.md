# Binsembler - A Binwise Ensembler

In general, Ensemble techniques combine the perspective of various models by aggregating the predictions output by each
of these models thus tend to enhance the overall prediction accuracy.  Simple techniques such as taking majority vote
or simple averaging of the predicted probabilities or weighted averaging of predicted probabilities based on model’s F1
score or Accuracy or any other measure are the popular choices to ensemble the model predictions. Here we propose a novel
approach based on aggregating the predicted probabilities as weighted averages where weights are the performance
statistic based on bins the probabilities fall in.

Idea is to divide the predicted probabilities of each model on a validation set into equal sized bins (preferably deciles)
and calculate the metrics in each bin. Pick any one metric, and note down it for each bin in a mapping table. This will be
the weight used in our weighted ensemble approach. When the prediction to be made on new data, first map the predicted
probabilities for the new data to an appropriate bin and then pick the respective metric value for that bin from the
mapping table and multiply with the predicted probability. Repeat the same for second model. Finally calculate a new
predicted probability as the weighted average.

## Processing Steps:

### Training

For each model repeat:

1. Create bins of equal size (say 10) of the predicted probabilities on the validation set
2. For each bin, calculate the Confusion matrix (TP, FP, TN, FN) and calculate other metrics such as Accuracy, AUC,
F1 score, Precision, Recall
3. Pick any metric (F1 Score) as the chosen weight
4. Store the chosen metric and corresponding bin information in a mapping table
5. Identify the Threshold that maximizes the chosen metric. This will be the Threshold used in our ensemble model

### Predict

1. Calculate the final probability for test observation as a weighted average of models probabilities
For e.g.:
Ensemble prob =  ((m1 F1 Score x m1 Predicted Probability) + (m2 F1 Score x m2 Predicted Probability))/((m1 F1 Score + m2 F1 Score))
2. If Ensemble prob > Threshold, then 1 else 0


## Ensemble Classifier

Ensemble for classification setting

## Run Book

```python
# Load the packages
from binsembler import Classifier

# Option 1: Providing Predicted Probabilities by each model

# Making up some probabilities
y1_preds = [0.25, 0.5, 0.75, 0.9]
y2_preds = [0.3, 0.6, 0.8, 0.9]
y1_acts = [0, 0, 1, 1]
y2_acts = [0, 1, 1, 1]

# Initialize Ensembler
cl = Classifier()

# Train our Ensembler
cl.train(y_pred_probs = [y1_preds, y2_preds], y_acts = [y1_acts, y2_acts])
print("Model Coefficients: ")
print(cl.w_)

# Predict using Ensembler
preds = cl.predict(y_pred_probs=[[0.87, 0.90, 0.1], [0.6, 0.7, 0.2]])
print(preds)
```

```python
# Option 2: Providing model and a training data

# create our first simple classification model using Naive Bayes
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Load dataset
data = load_breast_cancer()

# Organize our data
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

# Split our data
X_train, X_test, y_train, y_test = train_test_split(features,
                                                          labels,
                                                          test_size=0.33,
                                                          random_state=42)

# Initialize our classifier
gnb = GaussianNB()

# Train our classifier
model1 = gnb.fit(X_train, y_train)

# Create a second model using logistic regression
from sklearn.linear_model import LogisticRegression

# Initialize our classifier
logreg = LogisticRegression()

# Train our classifier
model2 = logreg.fit(X_train, y_train)

# Initialize Ensembler
cl = Classifier()

# Train our Ensembler
cl.train(models = [model1, model2], targetcols = ['target', 'target'], X_train=X_train, y_train=y_train)
print("Model Coefficients: ")
print(cl.w_[['m1_bins', 'm1_Accuracy', 'm2_Accuracy']])

# Predict using Ensembler
preds = cl.predict(models=[model1, model2], test_data=X_test)
print(preds.head())
```

## Building the package

Go tho the source folder where you have the `setup.py`. Run below command to build the package.

```sh
python setup.py sdist bdist_wheel
```

Once the package is built, you will see a `dist` folder and within the folder a `.tar.gz` file and `.whl` file. Run the
below command to install the package

```sh
python -m pip install name_of_the_whl_file.whl
```

You can also download the pre-built pacakge from `bin` folder and run the above command if you do not want to build.