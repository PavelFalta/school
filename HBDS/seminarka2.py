import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from bayes_opt import BayesianOptimization

def load_data(path):
    #load the specified data
    df = pd.read_csv(path)
    #print the first few rows of the dataframe
    print(df.head())
    return df

def preprocess_data(df, n_groups=2):
    #separate features and target variable
    X = df.drop(columns=["G1", "G2", "G3"])
    y = df["G3"]

    #encode categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {col: LabelEncoder().fit(X[col]) for col in categorical_cols}
    for col, le in label_encoders.items():
        X[col] = le.transform(X[col])

    #scale numerical features
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    #print summary statistics of the features
    print(X.describe().T)

    #ensure n_groups is at least 2
    if n_groups < 2:
        raise ValueError("n_groups must be at least 2")

    #bin the target variable into specified number of groups
    quantiles = [i / n_groups for i in range(1, n_groups)]
    bins = [-float('inf')] + y.quantile(quantiles).tolist() + [float('inf')]
    labels = list(range(n_groups))

    y = pd.cut(y, bins=bins, labels=labels).astype(int)
    #print the bins and value counts of the target variable
    print(bins, y.value_counts())

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

    #define cross-validation function for decision tree
    def dt_cv(max_depth, min_samples_split):
        model = DecisionTreeClassifier(max_depth=int(max_depth), min_samples_split=int(min_samples_split), random_state=42)
        return cross_val_score(model, X_train, y_train, scoring='accuracy', cv=5).mean()

    return rf_cv, svc_cv, knn_cv, xgb_cv, dt_cv

def optimize_model(cv_function, params):
    #optimize the model using bayesian optimization
    optimizer = BayesianOptimization(cv_function, params, random_state=42)
    optimizer.maximize(init_points=10, n_iter=30)
    return optimizer.max['params']

def main():
    path = 'HBDS/Datasets/Maths.csv'
    #path = 'HBDS/Datasets/Portuguese.csv'
    #load the data
    df = load_data(path)
    #preprocess the data
    X, y = preprocess_data(df, n_groups=2)
    #split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #define cross-validation functions for each model
    rf_cv, svc_cv, knn_cv, xgb_cv, dt_cv = define_cv_functions(X_train, y_train)

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

    dt_params_refined = {
        'max_depth': (1, 25),
        'min_samples_split': (2, 30)
    }

    #optimize each model
    rf_params = optimize_model(rf_cv, rf_params_refined)
    svc_params = optimize_model(svc_cv, svc_params_refined)
    knn_params = optimize_model(knn_cv, knn_params_refined)
    xgb_params = optimize_model(xgb_cv, xgb_params_refined)
    dt_params = optimize_model(dt_cv, dt_params_refined)

    #convert optimized parameters to appropriate types
    rf_params = {k: int(v) if k in ['n_estimators', 'max_depth', 'min_samples_split'] else v for k, v in rf_params.items()}
    svc_params = {k: v for k, v in svc_params.items()}
    knn_params = {k: int(v) for k, v in knn_params.items()}
    xgb_params = {k: int(v) if k in ['n_estimators', 'max_depth'] else v for k, v in xgb_params.items()}
    dt_params = {k: int(v) for k, v in dt_params.items()}

    #define base estimators for stacking classifier
    base_estimators = [
        ('rf', RandomForestClassifier(**rf_params, random_state=42)),
        ('svc', SVC(**svc_params, probability=True, random_state=42)),
        ('knn', KNeighborsClassifier(**knn_params)),
        ('xgb', XGBClassifier(**xgb_params, eval_metric="error", random_state=42))
    ]

    #define meta model for stacking classifier
    meta_model = DecisionTreeClassifier(**dt_params, random_state=42)

    #create and train stacking classifier
    stacking_clf = StackingClassifier(estimators=base_estimators, final_estimator=meta_model, cv=5)
    stacking_clf.fit(X_train, y_train)

    #make predictions and evaluate the model
    y_pred = stacking_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Stacking Model Accuracy: {accuracy:.2f}")

    #print refined best parameters for each model
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
    print("Decision Tree Parameters:")
    for param, value in dt_params.items():
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