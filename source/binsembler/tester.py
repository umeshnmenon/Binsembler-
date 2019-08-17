#from ensemble_classifier2 import EnsembleClassifier2
from classifier import Classifier
import sys


def main():
    # Option 1: Providing Predicted Probabilities by each model
    # Making up some probabilities
    y1_preds = [0.25, 0.5, 0.75, 0.9]
    y2_preds = [0.3, 0.6, 0.8, 0.9]
    y1_acts = [0, 0, 1, 1]
    y2_acts = [0, 1, 1, 1]

    # Initialize Ensembler
    cl = Classifier()

    # Train our Ensembler
    cl.train(y_pred_probs=[y1_preds, y2_preds], y_acts=[y1_acts, y2_acts])
    print("Model Coefficients: ")
    print(cl.w_)

    # Predict using Ensembler
    preds = cl.predict(y_pred_probs=[[0.87, 0.90, 0.1], [0.6, 0.7, 0.2]])
    print(preds)


    # Option 2: Providing model and a training data
    import sklearn
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    import pandas as pd

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
    cl.train(models=[model1, model2], targetcols=['target', 'target'], X_train=X_train, y_train=y_train)
    print("Model Coefficients: ")
    print(cl.w_[['m1_bins', 'm1_Accuracy', 'm2_Accuracy']])

    # Predict using Ensembler
    preds = cl.predict(y_pred_probs=[[0.87, 0.90, 0.1], [0.6, 0.7, 0.2]])
    print(preds)

if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nAborting ... Keyboard Interrupt.")
        sys.exit(1)
