from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, r2_score, mean_squared_error, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import joblib

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """ A majority vote ensemble classifier

    Parameters
    ----------
    classifiers : array-like, shape = [n_classifiers]
        Different classifiers for the ensemble

    vote : str, {'classlabel', 'probability'}
        Default: 'classlabel'
        If 'classlabel', the prediction is based on the argmax of class labels.
        Else if 'probability', the argmax of the sum of probabilities is used
        to predict the class label (recommended for calibrated classifiers).

    weights : array-like, shape = [n_classifiers]
        Optional, default: None
        If a list of `int` or `float` values are provided, the classifiers are
        weighted by importance; Uses uniform weights if `weights=None`.
    """

    def __init__(self, classifiers, vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def predict(self, X):
      """Predict class labels for X."""
      if self.vote == 'probability':
        maj_vote = np.argmax(self.predict_proba(X), axis=1)
      else:  # 'classlabel' vote
        predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
        maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.weights)), axis=1, arr=predictions)
      return self.lablenc_.inverse_transform(maj_vote)

    def predict_proba(self, X):
        """Predict class probabilities for X."""
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def fit(self, X, y):
        """ Fit classifiers.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Matrix of training samples.

        y : array-like, shape = [n_samples]
            Vector of target class labels.

        Returns
        -------
        self : object
        """
        # Use LabelEncoder to ensure class labels start with 0, which is important for np.argmax
        # call in self.predict
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self


def glass_pipeline(file_path, target_column):
    #load the dataset
    data = pd.read_csv(file_path)
    
    #task 1: data preprocessing
    if VERBOSE:
        #print initial data
        print("Initial Data:")
        print(data.head(), end='\n\n')

    #summarize missing values
    missing_summary = data.isnull().sum()
    if VERBOSE:
        #print missing values summary
        print("Missing Values Summary:")
        print(missing_summary, end='\n\n')

    #separate features and target variable
    X = data.drop(columns=target_column)
    y = data[target_column]
    le = LabelEncoder()
    y = le.fit_transform(y)

    if VERBOSE:
        #print encoded target variable
        print("Encoded Target Variable:")
        print(y, end='\n\n')

    #split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #task 2: select the base models
    classifiers = [
        ('Logistic Regression', LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state=42)),
        ('Decision Tree', DecisionTreeClassifier(max_depth=10, criterion='entropy', random_state=42)),
        ('KNN', KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')),
        ('Random Forest', RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=42))
    ]

    #create pipelines for each classifier
    pipelines = [(name, Pipeline([('sc', StandardScaler()), ('clf', clf)])) for name, clf in classifiers]

    #evaluate the models
    for name, pipeline in pipelines:
        if VERBOSE:
            #print evaluation message
            print(f"Evaluating {name}...")
        evaluate_model(pipeline, X_train, y_train, name)

    #task 3: combine the models using ensemble technique
    mv_clf = MajorityVoteClassifier(classifiers=[pipe for _, pipe in pipelines])
    if VERBOSE:
        #print evaluation message for majority voting classifier
        print("Evaluating Majority Voting Classifier...")
    evaluate_model(mv_clf, X_train, y_train, 'Majority Voting')
    print(end="\n\n")

    #weighted majority voting classifier
    mv_clf_weighted = MajorityVoteClassifier(classifiers=[pipe for _, pipe in pipelines], weights=[0.1, 0.2, 0.3, 0.4]) #since the random forest model is performing well, we give it most weight
    if VERBOSE:
        #print evaluation message for weighted majority voting classifier
        print("Evaluating Weighted Majority Voting Classifier...")
    evaluate_model(mv_clf_weighted, X_train, y_train, 'Majority Voting (Weighted)')
    print(end="\n\n")

    #task 4: compare the performance of your hybrid model with individual base models
    trained_clfs = []
    for name, pipeline in pipelines + [('Majority Voting', mv_clf), ('Majority Voting (Weighted)', mv_clf_weighted)]:
        if VERBOSE:
            #print training and saving message
            print(f"Training and saving {name}...")
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, f'{name}_model.pkl')
        trained_clfs.append(pipeline)
        evaluate_model(pipeline, X_train, y_train, name)

    #task 5: evaluate your model on test data
    for name, pipeline in pipelines + [('Majority Voting', mv_clf), ('Majority Voting (Weighted)', mv_clf_weighted)]:
        y_pred = pipeline.predict(X_test)
        if VERBOSE:
            #print evaluation message for test data
            print(f"Evaluation on test data for {name}:")
            print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
            print(f"F1 Score: {f1_score(y_test, y_pred, average='macro'):.3f}")
            print(confusion_matrix(y_test, y_pred), end='\n\n')

    #task 6: visualize the results
    if VERBOSE:
        #print visualizing results message
        print("Visualizing results...")
    visualize_results(pipelines + [('Majority Voting', mv_clf), ('Majority Voting (Weighted)', mv_clf_weighted)], X_test, y_test, le)


def breasts_pipeline(file_path, target_column):
    #load the dataset
    data = pd.read_csv(file_path)
    
    #task 1: data preprocessing
    if VERBOSE:
        #print initial data
        print("Initial Data:")
        print(data.head(), end='\n\n')

    #summarize missing values
    missing_summary = data.isnull().sum()
    if VERBOSE:
        #print missing values summary
        print(missing_summary, end='\n\n')

    #separate features and target variable
    X = data.drop(columns=target_column)
    y = data[target_column]
    le = LabelEncoder()
    y = le.fit_transform(y)

    if VERBOSE:
        #print encoded target variable
        print("Encoded Target Variable:")
        print(y, end='\n\n')

    #split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #task 2: select the base models
    classifiers = [
        ('Logistic Regression', LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state=42)),
        ('Decision Tree', DecisionTreeClassifier(max_depth=10, criterion='entropy', random_state=42)),
        ('KNN', KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')),
        ('Random Forest', RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=42))
    ]

    #create pipelines for each classifier
    pipelines = [(name, Pipeline([('sc', StandardScaler()), ('clf', clf)])) for name, clf in classifiers]

    #evaluate the models
    for name, pipeline in pipelines:
        if VERBOSE:
            #print evaluation message
            print(f"Evaluating {name}...")
        evaluate_model(pipeline, X_train, y_train, name, True)

    #task 3: combine the models using ensemble technique
    mv_clf = MajorityVoteClassifier(classifiers=[pipe for _, pipe in pipelines])
    if VERBOSE:
        #print evaluation message for majority voting classifier
        print("Evaluating Majority Voting Classifier...")
    evaluate_model(mv_clf, X_train, y_train, 'Majority Voting', True)
    print(end="\n\n")

    #weighted majority voting classifier
    mv_clf_weighted = MajorityVoteClassifier(classifiers=[pipe for _, pipe in pipelines], weights=[0.4, 0.1, 0.2, 0.2]) #since the logistic regression model is performing well, we give it most weight
    if VERBOSE:
        #print evaluation message for weighted majority voting classifier
        print("Evaluating Weighted Majority Voting Classifier...")
    evaluate_model(mv_clf_weighted, X_train, y_train, 'Majority Voting (Weighted)', True)
    print(end="\n\n")

    #task 4: compare the performance of your hybrid model with individual base models
    trained_clfs = []
    for name, pipeline in pipelines + [('Majority Voting', mv_clf), ('Majority Voting (Weighted)', mv_clf_weighted)]:
        if VERBOSE:
            #print training and saving message
            print(f"Training and saving {name}...")
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, f'{name}_model.pkl')
        trained_clfs.append(pipeline)
        evaluate_model(pipeline, X_train, y_train, name, True)

    #task 5: evaluate your model on test data
    for name, pipeline in pipelines + [('Majority Voting', mv_clf), ('Majority Voting (Weighted)', mv_clf_weighted)]:
        y_pred = pipeline.predict(X_test)
        if VERBOSE:
            #print evaluation message for test data
            print(f"Evaluation on test data for {name}:")
            print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
            print(f"F1 Score: {f1_score(y_test, y_pred, average='macro'):.3f}")
            print(f"ROC AUC: {roc_auc_score(y_test, y_pred):.3f}")
            print(confusion_matrix(y_test, y_pred), end='\n\n')

    #task 6: visualize the results
    if VERBOSE:
        #print visualizing results message
        print("Visualizing results...")
    visualize_results(pipelines + [('Majority Voting', mv_clf), ('Majority Voting (Weighted)', mv_clf_weighted)], X_test, y_test, le)

    #plot roc curve
    plot_roc_curve(pipelines + [('Majority Voting', mv_clf), ('Majority Voting (Weighted)', mv_clf_weighted)], X_test, y_test, le)

def evaluate_model(model, X_train, y_train, label, binary=False):
    #evaluate model accuracy
    scores_acc = cross_val_score(estimator=model, X=X_train, y=y_train, cv=6, scoring='accuracy')
    print("Accuracy: %0.3f (+/- %0.3f) [%s]" % (scores_acc.mean(), scores_acc.std(), label))
    
    #evaluate model f1 score
    scores_f1 = cross_val_score(estimator=model, X=X_train, y=y_train, cv=6, scoring='f1_macro')
    print("F1 Score: %0.3f (+/- %0.3f) [%s]" % (scores_f1.mean(), scores_f1.std(), label))
    
    if binary:
        #evaluate model roc auc score
        scores_roc_auc = cross_val_score(estimator=model, X=X_train, y=y_train, cv=6, scoring='roc_auc')
        print("ROC AUC: %0.3f (+/- %0.3f) [%s]" % (scores_roc_auc.mean(), scores_roc_auc.std(), label))

def plot_roc_curve(models, X_test, y_test, label_encoder):
    #plot roc curve for models
    plt.figure(figsize=(10, 8))
    for name, model in models:
        y_proba = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_proba):.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()
    
def visualize_results(models, X_test, y_test, label_encoder):
    #visualize model performance
    results = []
    for name, model in models:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        results.append((name, acc, f1))

    results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'F1 Score'])
    results_df.set_index('Model', inplace=True)
    print(results_df, end='\n\n')

    results_df[['Accuracy', 'F1 Score']].plot(kind='bar', figsize=(10, 6))
    plt.title('Model Performance')
    plt.ylabel('Score')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()


if __name__ == "__main__":

    VERBOSE = False

    glass_pipeline("Datasets/glass.csv", "Type")
    breasts_pipeline("Datasets/breast-cancer.csv", "diagnosis")