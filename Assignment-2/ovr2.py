from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
import pandas as pd
import logging
# Load data
data = pd.read_csv("ecommerceDataset.csv").dropna()
logging.basicConfig(filename="gridsearch_output2.log", level=logging.INFO, format="%(asctime)s - %(message)s")
data.columns = ['Category', 'Description']

# Split the data
vectorizer = CountVectorizer(max_features=5000)
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42, stratify=data['Category'])
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_data['Category'])

# Extract labels
y_train, y_val, y_test = train_data['Category'], val_data['Category'], test_data['Category']

# TF-IDF feature extraction (Avoid fitting on val/test to prevent leakage)
vectorizer = CountVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_data['Description'])
X_val = vectorizer.transform(val_data['Description'])
X_test = vectorizer.transform(test_data['Description'])

# Train the model
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

# Evaluate without leakage
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred, average='macro')
print("Test Macro F1 Score:", f1)
