import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from bayes_opt import BayesianOptimization
from sklearn.neural_network import MLPClassifier

def load_data(path):
    #load the specified data
    df = pd.read_csv(path)
    #print the first few rows of the dataframe
    print(df.head())
    return df

def preprocess_data(df):
    #separate features and target variable
    X = df.drop(columns=["benign_0__mal_1"])
    y = df["benign_0__mal_1"]

    #scale numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # print(pd.DataFrame(X).describe().T)

    return X, y

def define_cv_functions(X_train, y_train):
    #define cross-validation function for random forest
    def rf_cv(n_estimators, max_depth, min_samples_split):
        model = RandomForestClassifier(
            n_estimators=int(n_estimators), max_depth=int(max_depth), min_samples_split=int(min_samples_split), random_state=42
        )
        return cross_val_score(model, X_train, y_train, scoring='accuracy', cv=5).mean()

    #define cross-validation function for support vector classifier
    def svc_cv(C, gamma):
        model = SVC(C=C, gamma=gamma, probability=True, random_state=42)
        return cross_val_score(model, X_train, y_train, scoring='accuracy', cv=5).mean()

    #define cross-validation function for k-nearest neighbors
    def knn_cv(n_neighbors):
        model = KNeighborsClassifier(n_neighbors=int(n_neighbors))
        return cross_val_score(model, X_train, y_train, scoring='accuracy', cv=5).mean()

    #define cross-validation function for xgboost
    def xgb_cv(n_estimators, max_depth, learning_rate):
        model = XGBClassifier(
            n_estimators=int(n_estimators), max_depth=int(max_depth), learning_rate=learning_rate, eval_metric="error", random_state=42
        )
        return cross_val_score(model, X_train, y_train, scoring='accuracy', cv=5).mean()

    return rf_cv, svc_cv, knn_cv, xgb_cv

def optimize_model(cv_function, params):
    #optimize the model using bayesian optimization
    optimizer = BayesianOptimization(cv_function, params, random_state=42)
    optimizer.maximize(init_points=10, n_iter=30)
    return optimizer.max['params']

def meta_model(classifiers, X_train, X_test, y_train, y_test, mode="stacking"):
    if mode == "stacking":
        meta_model = StackingClassifier(estimators=classifiers, final_estimator=RandomForestClassifier(random_state=42))
    elif mode == "voting":
        meta_model = VotingClassifier(estimators=classifiers, voting='soft', weights=[0.3, 0.3, 0.1, 0.3])
    elif mode == "mlp":
        mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        meta_model = StackingClassifier(estimators=classifiers, final_estimator=mlp)

    meta_model.fit(X_train, y_train)
    y_pred = meta_model.predict(X_test)

    return y_pred

def main():
    path = 'HBDS/Datasets/cancer_classification.csv'
    optimize = False
    #load the data
    df = load_data(path)
    #preprocess the data
    X, y = preprocess_data(df)
    #split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #define cross-validation functions for each model
    rf_cv, svc_cv, knn_cv, xgb_cv = define_cv_functions(X_train, y_train)

    #define parameter ranges for optimization
    rf_params_refined = {
        'n_estimators': (2, 100),
        'max_depth': (1, 50),
        'min_samples_split': (2, 30)
    }

    svc_params_refined = {
        'C': (500, 1200),
        'gamma': (0.001, 0.2)
    }

    knn_params_refined = {
        'n_neighbors': (3, 50)
    }

    xgb_params_refined = {
        'n_estimators': (100, 300),
        'max_depth': (1, 25),
        'learning_rate': (0.01, 0.1)
    }

    #optimize each model
    if optimize:
        rf_params = optimize_model(rf_cv, rf_params_refined)
        svc_params = optimize_model(svc_cv, svc_params_refined)
        knn_params = optimize_model(knn_cv, knn_params_refined)
        xgb_params = optimize_model(xgb_cv, xgb_params_refined)
    else:
        rf_params = {'n_estimators': 19, 'max_depth': 41, 'min_samples_split': 7}
        svc_params = {'C': 609.1717792899823, 'gamma': 0.0090790881135151}
        knn_params = {'n_neighbors': 8}
        xgb_params = {'n_estimators': 136, 'max_depth': 6, 'learning_rate': 0.08491983767203796}

    #convert optimized parameters to appropriate types
    rf_params = {k: int(v) if k in ['n_estimators', 'max_depth', 'min_samples_split'] else v for k, v in rf_params.items()}
    svc_params = {k: v for k, v in svc_params.items()}
    knn_params = {k: int(v) for k, v in knn_params.items()}
    xgb_params = {k: int(v) if k in ['n_estimators', 'max_depth'] else v for k, v in xgb_params.items()}

    #define base estimators for stacking classifier
    base_estimators = [
        ('rf', RandomForestClassifier(**rf_params, random_state=42)),
        ('svc', SVC(**svc_params, probability=True, random_state=42)),
        ('knn', KNeighborsClassifier(**knn_params)),
        ('xgb', XGBClassifier(**xgb_params, eval_metric="error", random_state=42))
    ]

    y_pred = meta_model(base_estimators, X_train, X_test, y_train, y_test, mode="mlp")

    print(f"Voting Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Classification Report: {classification_report(y_test, y_pred)}")

    #print refined best parameters for each model
    if optimize:
        print("Refined Best Parameters for Each Model:")
        print("-" * 30)
        print("Random Forest Parameters:")
        for param, value in rf_params.items():
            print(f"  {param}: {value}")
        print("-" * 30)
        print("SVC Parameters:")
        for param, value in svc_params.items():
            print(f"  {param}: {value}")
        print("-" * 30)
        print("KNN Parameters:")
        for param, value in knn_params.items():
            print(f"  {param}: {value}")
        print("-" * 30)
        print("XGBoost Parameters:")
        for param, value in xgb_params.items():
            print(f"  {param}: {value}")
        print("-" * 30)

    #plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title("Stacking Model - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

if __name__ == "__main__":
    main()
