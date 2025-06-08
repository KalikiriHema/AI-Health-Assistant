# Step 1: Import Libraries
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Step 2: Load the Dataset
df = pd.read_csv("disease_symptom.csv")

# Step 3: Preprocess Symptoms
df['Symptom_List'] = df['Symptoms'].apply(lambda x: [s.strip().lower() for s in x.split(',')])

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
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Optional: View all known symptoms
print("\nSymptom Features Used:\n", mlb.classes_)

