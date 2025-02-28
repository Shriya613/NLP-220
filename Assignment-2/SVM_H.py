import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
import logging
# Load and preprocess data
data = pd.read_csv("ecommerceDataset.csv").dropna()
data.columns = ['Category', 'Description']

logging.basicConfig(filename="svm_output.log", level=logging.INFO, format="%(asctime)s - %(message)s")
# Train/Validation/Test Split
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42, shuffle=True)
val_data, test_data = train_test_split(temp_data, test_size=2/3, random_state=42, shuffle=True)
y_train = train_data['Category']

# Define feature extraction techniques
vectorizers = {
    "BoW": CountVectorizer(max_features=5000),
    "TF-IDF": TfidfVectorizer(max_features=5000),
    "N-grams": CountVectorizer(ngram_range=(1, 2), max_features=5000)
}

# Define parameters for GridSearch with LinearSVC
param_grid_linear_svc = {
    'C': [0.01, 0.1, 1, 10, 100],        # Regularization parameter
    #'tol': [ 1e-5]           # Tolerance values to explore
}

# Perform grid search function for LinearSVC
def perform_grid_search(X_train, model_name):
    grid_search = GridSearchCV(
        LinearSVC(),
        param_grid_linear_svc,
        scoring='f1_macro',
        cv=5,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    logging.info(f"Best SVM parameters for {model_name}: {best_params}")
    logging.info(f"Best F1 Macro for {model_name}: {best_score}")
    print(f"Best Linear SVC parameters for {model_name}:", best_params)
    print(f"Best F1 Macro for {model_name}:", best_score)
    return best_score

# Function to find and print the best feature engineering technique based on accuracy
def find_best_feature_engineering():
    best_score = 0
    best_feature = None
    
    for name, vectorizer in vectorizers.items():
        print(f"\n--- {name} ---")
        X_train = vectorizer.fit_transform(train_data['Description'])
        
        # Perform Grid Search
        score = perform_grid_search(X_train, name)
        
        # Update best feature engineering technique if current score is higher
        if score > best_score:
            best_score = score
            best_feature = name
    
    print(f"\nBest Feature Engineering Technique: {best_feature} with Accuracy: {best_score:.4f}")
    logging.info(f"Best Feature Engineering Technique: {best_feature} with Accuracy: {best_score:.4f}")

# Run the function to find the best feature engineering technique
find_best_feature_engineering()

