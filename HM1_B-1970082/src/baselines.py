import json
import random
from collections import Counter

import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# Load and preprocess data
def load_data(filepath):
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]
    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]
    return texts, labels


# Load pre-trained Word2Vec embeddings
def load_word2vec_model(filepath):
    # Load a Word2Vec model in KeyedVectors format
    return KeyedVectors.load_word2vec_format(filepath, binary=True)


# Average Word2Vec embeddings for each text
def average_word2vec_embedding(text, word2vec):
    vectors = [word2vec[word] for word in text.split() if word in word2vec]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(word2vec.vector_size)


# FFNN using averaged Word2Vec embeddings as input
def word2vec_ffnn_baseline(train_texts, train_labels, test_texts, word2vec):
    # Prepare averaged embeddings for train and test
    train_embeddings = np.array([average_word2vec_embedding(text, word2vec) for text in train_texts])
    test_embeddings = np.array([average_word2vec_embedding(text, word2vec) for text in test_texts])

    # Train a simple logistic regression as FFNN on the averaged embeddings
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(train_embeddings, train_labels)

    # Predict on test data
    predictions = classifier.predict(test_embeddings)
    return predictions

# Baseline functions
def random_baseline(test_data, label_choices):
    return [random.choice(label_choices) for _ in test_data]

def majority_baseline(train_labels, test_data):
    majority_label = Counter(train_labels).most_common(1)[0][0]
    return [majority_label] * len(test_data)

def stratified_baseline(train_labels, test_data):
    label_distribution = Counter(train_labels)
    total_count = sum(label_distribution.values())
    probabilities = {label: count / total_count for label, count in label_distribution.items()}
    return random.choices(
        population=list(probabilities.keys()),
        weights=list(probabilities.values()),
        k=len(test_data)
    )

# Evaluation function
# Evaluation function
def evaluate_baseline(predictions, true_labels):
    """Calculates accuracy, precision, recall, and F1-score for the predictions."""
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    return accuracy, precision, recall, f1


# Main execution
if __name__ == "__main__":
    # Load training and validation data
    train_texts, train_labels = load_data('../data/train-taskA.jsonl')
    val_texts, val_labels = load_data('../data/test-news-taskA.jsonl')  # Change as needed

    label_choices = list(set(train_labels))

    # Random Baseline
    print("Random Baseline Results:")
    random_predictions = random_baseline(val_labels, label_choices)
    evaluate_baseline(random_predictions, val_labels)

    # Majority Baseline
    print("\nMajority Baseline Results:")
    majority_predictions = majority_baseline(train_labels, val_labels)
    evaluate_baseline(majority_predictions, val_labels)

    # Stratified Baseline
    print("\nStratified Baseline Results:")
    stratified_predictions = stratified_baseline(train_labels, val_labels)
    evaluate_baseline(stratified_predictions, val_labels)

    # Word2Vec + FFNN Baseline
    print("\nWord2Vec + FFNN Baseline Results:")
    word2vec_path = 'D:/Repos/MNLP/mnlp-hws/mnlp-hw1b/HM1_B-1970082/data/GoogleNews-vectors-negative300.bin'
    word2vec_model = load_word2vec_model(word2vec_path)
    word2vec_ffnn_predictions = word2vec_ffnn_baseline(train_texts, train_labels, val_texts, word2vec_model)
    evaluate_baseline(word2vec_ffnn_predictions, val_labels)
