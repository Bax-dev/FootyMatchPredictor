# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from the Excel file
file_path = 'epl_match_data.xlsx'  # Replace with your actual file path
data = pd.read_excel(file_path)

# Explore the dataset (optional)
print(data.head())
print(data.describe())
print(data.isnull().sum())

# Encode categorical variables (home_team, away_team, and result)
label_encoder = LabelEncoder()
data['home_team'] = label_encoder.fit_transform(data['home_team'])
data['away_team'] = label_encoder.fit_transform(data['away_team'])
data['result'] = label_encoder.fit_transform(data['result'])

# Fill missing values if necessary (optional, depending on your dataset)
data.fillna(0, inplace=True)

# Define features (X) and target (y)
X = data[['home_team', 'away_team', 'home_goals', 'away_goals']]
y = data['result']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the trained model to a file (optional)
joblib.dump(model, 'football_match_predictor.pkl')

# Plotting the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plotting Feature Importances
importances = model.feature_importances_
features = X.columns
indices = range(len(importances))

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features, palette="viridis")
plt.title("Feature Importances")
plt.show()

# Plotting Count Plot of True vs Predicted Labels
plt.figure(figsize=(12, 6))

# True labels count plot
plt.subplot(1, 2, 1)
sns.countplot(x=y_test, palette="viridis")
plt.title('True Labels Distribution')
plt.xlabel('Match Outcome')
plt.ylabel('Count')
plt.xticks(ticks=range(len(label_encoder.classes_)), labels=label_encoder.classes_)

# Predicted labels count plot
plt.subplot(1, 2, 2)
sns.countplot(x=y_pred, palette="viridis")
plt.title('Predicted Labels Distribution')
plt.xlabel('Match Outcome')
plt.ylabel('Count')
plt.xticks(ticks=range(len(label_encoder.classes_)), labels=label_encoder.classes_)

plt.tight_layout()
plt.show()
