import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('Datasets/glass.csv')

X = data.drop(columns=['Type'])
y = data['Type']

X_scaled = StandardScaler().fit_transform(X)
y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, bootstrap=True, max_depth=10, max_features='sqrt', min_samples_leaf=1, min_samples_split=10, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Classification Report for Random Forest:")
print(classification_report(y_test, y_pred_rf, zero_division=0))

dt = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=10, min_samples_leaf=1, random_state=42)
dt.fit(X_train, y_train)

bagging_dt = BaggingClassifier(estimator=dt, n_estimators=10, random_state=42, n_jobs=-1)
bagging_dt.fit(X_train, y_train)
y_pred_bagging_dt = bagging_dt.predict(X_test)
print("Bagging DT:")
print(classification_report(y_test, y_pred_bagging_dt, zero_division=0))

bagging_rf = BaggingClassifier(estimator=rf, n_estimators=10, random_state=42, n_jobs=-1)
bagging_rf.fit(X_train, y_train)
y_pred_bagging_rf = bagging_rf.predict(X_test)
print("Bagging RF:")
print(classification_report(y_test, y_pred_bagging_rf, zero_division=0))

kn = KNeighborsClassifier(n_neighbors=3, p=2)
kn.fit(X_train, y_train)
bagging_kn = BaggingClassifier(estimator=kn, n_estimators=10, random_state=42, n_jobs=-1)
bagging_kn.fit(X_train, y_train)
y_pred_bagging_kn = bagging_kn.predict(X_test)
print("Bagging KNN:")
print(classification_report(y_test, y_pred_bagging_kn, zero_division=0))

ada_dt = AdaBoostClassifier(estimator=dt, n_estimators=500, learning_rate=0.1, random_state=42)
ada_dt.fit(X_train, y_train)
y_pred_ada_dt = ada_dt.predict(X_test)
print("AdaBoost DT:")
print(classification_report(y_test, y_pred_ada_dt, zero_division=0))

ada_rf = AdaBoostClassifier(estimator=rf, n_estimators=50, learning_rate=0.1, random_state=42)
ada_rf.fit(X_train, y_train)
y_pred_ada_rf = ada_rf.predict(X_test)
print("AdaBoost RF:")
print(classification_report(y_test, y_pred_ada_rf, zero_division=0))

gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, max_depth=10, min_samples_split=4, min_samples_leaf=1, subsample=1.0, max_features=None, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
print("Gradient Boosting:")
print(classification_report(y_test, y_pred_gb, zero_division=0))
