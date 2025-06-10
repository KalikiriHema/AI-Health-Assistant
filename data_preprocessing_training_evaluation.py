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
import os # Import the os module
import seaborn as sns # Import seaborn here

# Check the current working directory to help locate the file
current_directory = os.getcwd()
print("Current working directory:", current_directory)

# Step 2: Load the Dataset
# Ensure 'disease_symptom.csv' is in the current directory or provide the full path
file_path = "/comprehensive_disease_data.csv"  # Using relative path first

# Check if the file exists before trying to read it
if not os.path.exists(file_path):
    # If the file is not found in the current directory, you might need to
    # update 'file_path' with the correct path to your file.
    # Example: file_path = "/path/to/your/data/disease_symptom.csv"
    print(f"Error: The file '{file_path}' was not found.")
    print(f"Please ensure '{file_path}' is in the current directory or provide the correct path.")
    # You might want to exit the script or handle this error appropriately
    # For now, we'll raise the error again to stop execution
    raise FileNotFoundError(f"[Errno 2] No such file or directory: '{file_path}'")


df = pd.read_csv(file_path)

# Step 3: Process Symptoms Correctly
def process_symptoms(symptoms):
    if isinstance(symptoms, list):
        return [s.strip().lower() for s in symptoms]
    # Handle potential NaN values or non-string types gracefully
    if isinstance(symptoms, str):
        return [s.strip().lower() for s in symptoms.split(',')]
    # Return an empty list or a default value if the symptom entry is not a string or list
    return []


df['Symptom_List'] = df['Symptoms'].apply(process_symptoms)

# Step 4: Filter Top 10 Diseases
top_diseases = df['Disease'].value_counts().head(10).index.tolist()
filtered_df = df[df['Disease'].isin(top_diseases)].copy()

# Step 5: Convert Symptoms to Binary Features
mlb = MultiLabelBinarizer()
# Ensure that 'Symptom_List' column doesn't contain None or NaN values
filtered_df = filtered_df[filtered_df['Symptom_List'].notna()]
X = mlb.fit_transform(filtered_df['Symptom_List'])
y = filtered_df['Disease']

# Step 6: Train-Test Split
# Check if there are enough samples for splitting
if len(filtered_df) < 2:
     print("Error: Not enough data after filtering to perform train-test split.")
     # Handle this case appropriately, e.g., exit or skip model training
else:
    # Removed stratify=y because some classes have only 1 sample in the filtered data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 7: Train the Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 8: Make Predictions and Evaluate
    y_pred = model.predict(X_test)
    print("📋 Classification Report:\n")
    # Ensure labels are passed to classification_report if needed for completeness
    print(classification_report(y_test, y_pred, zero_division=0))


    # Step 9: Confusion Matrix (using ConfusionMatrixDisplay)
    # Ensure labels are consistent
    # Use labels present in both y_test and y_pred for the ConfusionMatrixDisplay
    display_labels = sorted(list(set(y_test) | set(y_pred)))
    cm = confusion_matrix(y_test, y_pred, labels=display_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    plt.figure(figsize=(10, 6))
    disp.plot(xticks_rotation=90, cmap="Blues")
    plt.title("Confusion Matrix - Disease Prediction (ConfusionMatrixDisplay)")
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
    print(comparison_df.sample(min(10, len(comparison_df)), random_state=42)) # Sample max of 10 or the total number of rows

    # Step 12: Confusion Matrix using Seaborn
    # y_test: true labels, y_pred: model predictions
    unique_labels = sorted(list(set(y_test) | set(y_pred)))
    # Ensure all model classes are included for the confusion matrix labels
    all_labels = sorted(list(set(y_test) | set(y_pred) | set(model.classes_)))
    cm_seaborn = confusion_matrix(y_test, y_pred, labels=all_labels)

    # Plot setup
    plt.figure(figsize=(12, 10)) # Increased figure size
    sns.heatmap(cm_seaborn, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=all_labels, yticklabels=all_labels)

    # Titles and axis labels
    plt.title('Confusion Matrix - Disease Prediction (Seaborn)', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    # Aesthetic adjustments
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save image (optional)
    # plt.savefig('confusion_matrix_disease_prediction_seaborn.png', dpi=300)
    plt.show() # Display the plot
