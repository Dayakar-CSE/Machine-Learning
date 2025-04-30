print("NAIVE BAYES ENGLISH TEXT CLASSIFICATION")

# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score

# Set seaborn style for plots
sns.set()

# Load all 20 categories
data = fetch_20newsgroups()
text_categories = data.target_names

# Load training and test data
train_data = fetch_20newsgroups(subset="train", categories=text_categories)
test_data = fetch_20newsgroups(subset="test", categories=text_categories)

# Print dataset statistics
print(f"We have {len(text_categories)} unique classes")
print(f"We have {len(train_data.data)} training samples")
print(f"We have {len(test_data.data)} test samples")

# Build and train the Naïve Bayes model pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train_data.data, train_data.target)

# Predict categories of the test data
predicted_categories = model.predict(test_data.data)

# Print sample predicted categories (optional)
print("Sample predictions:", np.array(test_data.target_names)[predicted_categories[:5]])

# Generate and plot the confusion matrix
mat = confusion_matrix(test_data.target, predicted_categories)
plt.figure(figsize=(10, 10))
sns.heatmap(mat.T, square=True, annot=True, fmt="d",
            xticklabels=train_data.target_names,
            yticklabels=train_data.target_names, cmap="Blues")

plt.xlabel("True Label")
plt.ylabel("Predicted Label")
plt.title("Confusion Matrix - Naïve Bayes Text Classification")
plt.tight_layout()
plt.show()

# Print accuracy
print(f"The accuracy is {accuracy_score(test_data.target, predicted_categories):.4f}")
