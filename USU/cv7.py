import openai
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import numpy as np

import openai

openai.api_key = "nice try diddy"
path = "data/kopisty_pocasi_rozsireno.csv"

with open(path, 'r') as file:
    reader = csv.reader(file, delimiter=';')
    
    df = pd.read_csv(file, delimiter=';', decimal=',')

print(df.head())

df['day_in_year'] = df['datum'].apply(lambda x: datetime.datetime.strptime(x, '%d.%m.%Y').strftime('%j'))

#srazky should be mean of uhrn_srazky_1 and uhrn_srazky_2

df.drop(columns=['datum', 'vypar'], inplace=True)

df['srazky'] = df[['uhrn_srazky_1', 'uhrn_srazky_2']].astype(float).mean(axis=1)

df.drop(columns=['uhrn_srazky_1', 'uhrn_srazky_2'], inplace=True)

# corr_matrix = df.corr()

# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
# plt.title('Correlation Matrix')
# plt.tight_layout()
# plt.show()

Y = df['srazky']
X = df.drop(columns=['srazky'])

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Prepare data for OpenAI
test_data = []
feature_names = df.drop(columns=['srazky']).columns.tolist()
for i in range(len(X_test)):
    features = X_test[i]
    # Create a descriptive prompt with feature names and values
    feature_desc = ", ".join([f"{name}: {val:.2f}" for name, val in zip(feature_names, features)])
    prompt = f"""Based on these weather features, predict if it will rain (1) or not rain (0). 
    Features: {feature_desc}
    Respond with only 0 or 1."""
    test_data.append(prompt)

# Get predictions from OpenAI
openai_predictions = []
for prompt in test_data:
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1  # Low temperature for more consistent binary outputs
        )
        # Extract the prediction (0 or 1) from the response
        prediction = int(response.choices[0].message.content.strip())
        openai_predictions.append(prediction)
    except Exception as e:
        print(f"Error getting prediction: {e}")
        openai_predictions.append(0)  # Default to 0 in case of error

# Convert to numpy array for comparison
openai_predictions = np.array(openai_predictions)

# Create binary labels for actual values
actual_rain = Y_test.copy()
actual_rain[actual_rain > 0] = 1
actual_rain[actual_rain <= 0] = 0

# Calculate confusion matrix for OpenAI predictions
cm_openai = confusion_matrix(actual_rain, openai_predictions)
print("\nOpenAI Confusion Matrix:")
print(cm_openai)
print("\nOpenAI Classification Report:")
print(classification_report(actual_rain, openai_predictions))

# Plot OpenAI predictions
plt.figure(figsize=(14, 7))
plt.plot(range(len(Y_test)), openai_predictions, label='OpenAI Predicted Rain', color='green', linewidth=2)
plt.plot(range(len(Y_test)), actual_rain, label='Actual Rain', color='red', linewidth=2)
plt.ylim(-0.2, 1.2)
plt.yticks([0, 1], ['No Rain', 'Rain'])
plt.xlabel('Time Index')
plt.title('OpenAI Rain/No-Rain Classification')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Display confusion matrix for OpenAI
disp_openai = ConfusionMatrixDisplay(confusion_matrix=cm_openai, display_labels=['No Rain', 'Rain'])
disp_openai.plot()
plt.title('OpenAI Confusion Matrix')
plt.show()

