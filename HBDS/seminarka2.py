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
    df = pd.read_csv(path)
    print(df.head())
    return df

def preprocess_data(df, n_groups=2):
    X = df.drop(columns=["G1", "G2", "G3"])
    y = df["G3"]

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {col: LabelEncoder().fit(X[col]) for col in categorical_cols}
    for col, le in label_encoders.items():
        X[col] = le.transform(X[col])

    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    print(X.describe().T)

    if n_groups < 2:
        raise ValueError("n_groups must be at least 2")

    quantiles = [i / n_groups for i in range(1, n_groups)]
    bins = [-float('inf')] + y.quantile(quantiles).tolist() + [float('inf')]
    labels = list(range(n_groups))

    y = pd.cut(y, bins=bins, labels=labels).astype(int)
    print(bins,y.value_counts())

    return X, y

def define_cv_functions(X_train, y_train):
    def rf_cv(n_estimators, max_depth, min_samples_split):
        model = RandomForestClassifier(
            n_estimators=int(n_estimators), max_depth=int(max_depth), min_samples_split=int(min_samples_split), random_state=42
        )
        return cross_val_score(model, X_train, y_train, scoring='accuracy', cv=5).mean()

    def svc_cv(C, gamma):
        model = SVC(C=C, gamma=gamma, probability=True, random_state=42)
        return cross_val_score(model, X_train, y_train, scoring='accuracy', cv=5).mean()

    def knn_cv(n_neighbors):
        model = KNeighborsClassifier(n_neighbors=int(n_neighbors))
        return cross_val_score(model, X_train, y_train, scoring='accuracy', cv=5).mean()

    def xgb_cv(n_estimators, max_depth, learning_rate):
        model = XGBClassifier(
            n_estimators=int(n_estimators), max_depth=int(max_depth), learning_rate=learning_rate, eval_metric="error", random_state=42
        )
        return cross_val_score(model, X_train, y_train, scoring='accuracy', cv=5).mean()

    def dt_cv(max_depth, min_samples_split):
        model = DecisionTreeClassifier(max_depth=int(max_depth), min_samples_split=int(min_samples_split), random_state=42)
        return cross_val_score(model, X_train, y_train, scoring='accuracy', cv=5).mean()

    return rf_cv, svc_cv, knn_cv, xgb_cv, dt_cv

def optimize_model(cv_function, params):
    optimizer = BayesianOptimization(cv_function, params, random_state=42)
    optimizer.maximize(init_points=10, n_iter=30)
    return optimizer.max['params']

def main():
    path = 'HBDS/Datasets/Maths.csv'
    # path = 'HBDS/Datasets/Portuguese.csv'
    df = load_data(path)
    X, y = preprocess_data(df, n_groups=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf_cv, svc_cv, knn_cv, xgb_cv, dt_cv = define_cv_functions(X_train, y_train)

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

    rf_params = optimize_model(rf_cv, rf_params_refined)
    svc_params = optimize_model(svc_cv, svc_params_refined)
    knn_params = optimize_model(knn_cv, knn_params_refined)
    xgb_params = optimize_model(xgb_cv, xgb_params_refined)
    dt_params = optimize_model(dt_cv, dt_params_refined)

    rf_params = {k: int(v) if k in ['n_estimators', 'max_depth', 'min_samples_split'] else v for k, v in rf_params.items()}
    svc_params = {k: v for k, v in svc_params.items()}
    knn_params = {k: int(v) for k, v in knn_params.items()}
    xgb_params = {k: int(v) if k in ['n_estimators', 'max_depth'] else v for k, v in xgb_params.items()}
    dt_params = {k: int(v) for k, v in dt_params.items()}

    base_estimators = [
        ('rf', RandomForestClassifier(**rf_params, random_state=42)),
        ('svc', SVC(**svc_params, probability=True, random_state=42)),
        ('knn', KNeighborsClassifier(**knn_params)),
        ('xgb', XGBClassifier(**xgb_params, eval_metric="error", random_state=42))
    ]

    meta_model = DecisionTreeClassifier(**dt_params, random_state=42)

    stacking_clf = StackingClassifier(estimators=base_estimators, final_estimator=meta_model, cv=5)
    stacking_clf.fit(X_train, y_train)

    y_pred = stacking_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Stacking Model Accuracy: {accuracy:.2f}")

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
