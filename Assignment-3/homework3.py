import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
import time

# Step 1: Load dataset
file_path = 'arxiv_data.json'
df = pd.read_json(file_path)

# Rename columns
df.columns = ['title', 'abstract', 'labels']

# Drop rows with missing values
df = df.dropna(subset=['abstract', 'labels'])

# Step 2: Dataset split
train, valtest = train_test_split(df, test_size=0.30, random_state=1234)
val, test = train_test_split(valtest, test_size=0.50, random_state=1234)

# Step 3: Encode labels using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train['labels'])
y_val = mlb.transform(val['labels'])
y_test = mlb.transform(test['labels'])

# Step 4: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train = vectorizer.fit_transform(train['abstract'])
X_val = vectorizer.transform(val['abstract'])
X_test = vectorizer.transform(test['abstract'])

# Step 5: Initialize and train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

print("\nTraining Random Forest...")
start_train_time = time.time()
rf.fit(X_train, y_train)
training_time = time.time() - start_train_time

# Step 6: Predict on validation set
print("\nPredicting on validation set...")
start_inference_time = time.time()
y_val_pred = rf.predict(X_val)
inference_time = time.time() - start_inference_time

# Step 7: Validation metrics
val_accuracy = accuracy_score(y_val, y_val_pred)
val_micro_f1 = f1_score(y_val, y_val_pred, average='micro')
val_macro_f1 = f1_score(y_val, y_val_pred, average='macro')
val_report = classification_report(y_val, y_val_pred, target_names=mlb.classes_, zero_division=0, output_dict=True)

# Extract specific classes (e.g., majority and minority)
majority_class = 'cs.CV'
minority_class = 'math.SP'

majority_metrics = val_report.get(majority_class, {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0})
minority_metrics = val_report.get(minority_class, {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0})

# Extract macro and weighted averages
val_macro_avg = val_report['macro avg']
val_weighted_avg = val_report['weighted avg']

# Display validation classification report
print("\nValidation Classification Report:")
print(f"    {majority_class}       {majority_metrics['precision']:.2f}      {majority_metrics['recall']:.2f}      {majority_metrics['f1-score']:.2f}      {majority_metrics['support']}")
print(f"    {minority_class}       {minority_metrics['precision']:.2f}      {minority_metrics['recall']:.2f}      {minority_metrics['f1-score']:.2f}      {minority_metrics['support']}")
print(f"    accuracy                           {val_accuracy:.2f}")
print(f"   macro avg       {val_macro_avg['precision']:.2f}      {val_macro_avg['recall']:.2f}      {val_macro_avg['f1-score']:.2f}")
print(f"weighted avg       {val_weighted_avg['precision']:.2f}      {val_weighted_avg['recall']:.2f}      {val_weighted_avg['f1-score']:.2f}")

# Step 8: Predict on test set
print("\nPredicting on test set...")
y_test_pred = rf.predict(X_test)

# Test metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_micro_f1 = f1_score(y_test, y_test_pred, average='micro')
test_macro_f1 = f1_score(y_test, y_test_pred, average='macro')
test_report = classification_report(y_test, y_test_pred, target_names=mlb.classes_, zero_division=0, output_dict=True)

# Extract specific classes (e.g., majority and minority)
majority_metrics_test = test_report.get(majority_class, {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0})
minority_metrics_test = test_report.get(minority_class, {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0})

# Extract macro and weighted averages
test_macro_avg = test_report['macro avg']
test_weighted_avg = test_report['weighted avg']

# Display test classification report
print("\nTest Classification Report:")
print(f"    {majority_class}       {majority_metrics_test['precision']:.2f}      {majority_metrics_test['recall']:.2f}      {majority_metrics_test['f1-score']:.2f}      {majority_metrics_test['support']}")
print(f"    {minority_class}       {minority_metrics_test['precision']:.2f}      {minority_metrics_test['recall']:.2f}      {minority_metrics_test['f1-score']:.2f}      {minority_metrics_test['support']}")
print(f"    accuracy                           {test_accuracy:.2f}")
print(f"   macro avg       {test_macro_avg['precision']:.2f}      {test_macro_avg['recall']:.2f}      {test_macro_avg['f1-score']:.2f}")
print(f"weighted avg       {test_weighted_avg['precision']:.2f}      {test_weighted_avg['recall']:.2f}      {test_weighted_avg['f1-score']:.2f}")

# Print timing metrics
print(f"\nModel Training Time: {training_time:.2f} seconds")
print(f"Model Inference Time (Validation): {inference_time:.2f} seconds")

output_file = "results.txt"

with open(output_file, "w") as f:
    # Write validation report
    f.write("Validation Classification Report:\n")
    f.write(f"    {majority_class}       {majority_metrics['precision']:.2f}      {majority_metrics['recall']:.2f}      {majority_metrics['f1-score']:.2f}      {majority_metrics['support']}\n")
    f.write(f"    {minority_class}       {minority_metrics['precision']:.2f}      {minority_metrics['recall']:.2f}      {minority_metrics['f1-score']:.2f}      {minority_metrics['support']}\n")
    f.write(f"    accuracy                           {val_accuracy:.2f}\n")
    f.write(f"   macro avg       {val_macro_avg['precision']:.2f}      {val_macro_avg['recall']:.2f}      {val_macro_avg['f1-score']:.2f}\n")
    f.write(f"weighted avg       {val_weighted_avg['precision']:.2f}      {val_weighted_avg['recall']:.2f}      {val_weighted_avg['f1-score']:.2f}\n\n")

    # Write test report
    f.write("Test Classification Report:\n")
    f.write(f"    {majority_class}       {majority_metrics_test['precision']:.2f}      {majority_metrics_test['recall']:.2f}      {majority_metrics_test['f1-score']:.2f}      {majority_metrics_test['support']}\n")
    f.write(f"    {minority_class}       {minority_metrics_test['precision']:.2f}      {minority_metrics_test['recall']:.2f}      {minority_metrics_test['f1-score']:.2f}      {minority_metrics_test['support']}\n")
    f.write(f"    accuracy                           {test_accuracy:.2f}\n")
    f.write(f"   macro avg       {test_macro_avg['precision']:.2f}      {test_macro_avg['recall']:.2f}      {test_macro_avg['f1-score']:.2f}\n")
    f.write(f"weighted avg       {test_weighted_avg['precision']:.2f}      {test_weighted_avg['recall']:.2f}      {test_weighted_avg['f1-score']:.2f}\n\n")

    # Timing metrics
    f.write(f"Model Training Time: {training_time:.2f} seconds\n")
    f.write(f"Model Inference Time (Validation): {inference_time:.2f} seconds\n")

print(f"Results saved to {output_file}")
