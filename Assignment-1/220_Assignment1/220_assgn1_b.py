import os
import pandas as pd


# Function to parse reviews and their sentiment labels from the given directory
def parse_imdb_reviews(dataset_dir, split):
    data = []
    
    # Paths to pos and neg subdirectories
    for sentiment in ['pos', 'neg']:
        sentiment_dir = os.path.join(dataset_dir, split, sentiment)
        label = 1 if sentiment == 'pos' else 0  # 1 for positive, 0 for negative
        
        # Loop through all files in the sentiment directory
        for file_name in os.listdir(sentiment_dir):
            if file_name.endswith('.txt'):
                file_path = os.path.join(sentiment_dir, file_name)
                
                # Extract the review text
                with open(file_path, 'r', encoding='utf-8') as f:
                    review_text = f.read()
                
                # Append review text and label to the data
                data.append([review_text, label, split])
    
    return pd.DataFrame(data, columns=['review', 'label', 'split'])

# Assuming you have the dataset at ./aclImdb
dataset_dir = "./aclImdb"

# Step 1: Parse the train and test datasets
train_data = parse_imdb_reviews(dataset_dir, 'train')
test_data = parse_imdb_reviews(dataset_dir, 'test')

# Combine train and test data into a single DataFrame
imdb_reviews_df = pd.concat([train_data, test_data], ignore_index=True)

# Save the parsed dataset as CSV
csv_path = './imdb_reviews.csv'
imdb_reviews_df.to_csv(csv_path, index=False)

print(f"Dataset saved to {csv_path}")

import matplotlib.pyplot as plt
import seaborn as sns

# Plot class distribution for training data
sns.countplot(data=train_data, x='label', palette='viridis')
plt.title('Class Distribution in Training Data')
plt.xlabel('Sentiment (0 = Negative, 1 = Positive)')
plt.ylabel('Number of Reviews')
plt.show()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

# Step 3: Load the dataset from the CSV
df = pd.read_csv('./imdb_reviews.csv')

# Step 4: Split the data into training and validation sets (train set: 90%, validation: 10%)
train_data, val_data = train_test_split(df[df['split'] == 'train'], test_size=0.1, random_state=42)
test_data = df[df['split'] == 'test']

# Step 5: N-gram feature extraction using CountVectorizer
vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english', max_features=5000,min_df=3)
X_train = vectorizer.fit_transform(train_data['review'])
X_val = vectorizer.transform(val_data['review'])
X_test = vectorizer.transform(test_data['review'])

y_train = train_data['label']
y_val = val_data['label']
y_test = test_data['label']

# Step 6: Define models for training and evaluation
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(solver = 'saga', max_iter=300),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC()
}

validation_accuracies = {}
test_accuracies = {}

# Step 7: Train and evaluate the models
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Calculate validation and test accuracies
    val_acc = accuracy_score(y_val, y_val_pred)
    validation_accuracies[model_name] = val_acc
    
    test_acc = accuracy_score(y_test, y_test_pred)
    test_accuracies[model_name] = test_acc
    
    print(f"{model_name}: Validation Accuracy = {val_acc:.4f}, Test Accuracy = {test_acc:.4f}")

# Step 8: Plotting the validation and test accuracies
plt.figure(figsize=(10, 6))
plt.bar(validation_accuracies.keys(), validation_accuracies.values(), color='b', label='Validation Accuracy')
plt.bar(test_accuracies.keys(), test_accuracies.values(), color='r', label='Test Accuracy', alpha=0.7)
plt.title('Model Accuracies')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.legend()
plt.show()

