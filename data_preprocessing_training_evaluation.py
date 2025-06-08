# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Step 2: Load the Dataset
df = pd.read_csv("disease_symptom.csv")

# Step 3: Process Symptoms Correctly
def process_symptoms(symptoms):
    if isinstance(symptoms, list):
        return [s.strip().lower() for s in symptoms]
    return [s.strip().lower() for s in symptoms.split(',')]

df['Symptom_List'] = df['Symptoms'].apply(process_symptoms)

# Step 4: Filter Top 10 Diseases
top_diseases = df['Disease'].value_counts().head(10).index.tolist()
filtered_df = df[df['Disease'].isin(top_diseases)].copy()

# Step 5: Convert Symptoms to Binary Features
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(filtered_df['Symptom_List'])
y = filtered_df['Disease']

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Make Predictions and Evaluate
y_pred = model.predict(X_test)
print("📋 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Step 9: Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

plt.figure(figsize=(10, 6))
disp.plot(xticks_rotation=90, cmap="Blues")
plt.title("Confusion Matrix - Disease Prediction")
plt.tight_layout()
plt.show()

# Step 10: Bar Plot of Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

metrics = [accuracy, precision, recall, f1]
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

plt.figure(figsize=(8, 5))
plt.bar(labels, metrics, color=['skyblue', 'lightgreen', 'orange', 'violet'])
plt.ylim(0, 1)
plt.title('Model Performance Metrics')
plt.ylabel('Score')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# Step 11: Show Sample Predictions
comparison_df = pd.DataFrame({
    "Actual Disease": y_test,
    "Predicted Disease": y_pred
})
print("\n🔍 Sample Predictions:")
print(comparison_df.sample(10, random_state=42))
