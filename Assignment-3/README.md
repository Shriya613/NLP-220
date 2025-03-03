## Assignment - 3 ##

This project focuses on multi-label classification of academic paper abstracts from the ArXiv dataset. The task involves predicting the appropriate labels (e.g., cs.CV, stat.ML) for each paper based on its abstract.

The dataset was split into training (70%), validation (15%), and test (15%) sets to evaluate model performance effectively.

Best Performing Model

Model Type: Random Forest Classifier

Key Results:
Validation Micro F1: 0.8136
Validation Macro F1: 0.4319
Model Training Time: 51.52 seconds
Model Inference Time: 0.94 seconds

Data Preprocessing
Removing Missing Values:

Rows with missing values in the abstract or labels columns were dropped.
TF-IDF Vectorization: The abstract column was vectorized using TF-IDF with a maximum of 5000 features. Stopwords were removed during vectorization to reduce noise.

Label Encoding:The labels column was binarized using MultiLabelBinarizer to handle the multi-label nature of the task.

Top Label Distribution:The top 10 most frequent labels in the training set were analyzed and plotted to understand the dataset's imbalance.

To run: 
$ python homework3.py --data "arxiv_data.json" --output "results.txt"

$ cat results.txt
