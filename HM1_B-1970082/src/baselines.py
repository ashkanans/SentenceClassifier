import json
import random
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Load and preprocess data
def load_data(filepath):
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]
    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]
    return texts, labels


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
def evaluate_baseline(predictions, true_labels):
    return accuracy_score(true_labels, predictions)


# Main execution
if __name__ == "__main__":
    # Load training and validation data
    train_texts, train_labels = load_data('../data/train-taskA.jsonl')
    _, val_labels = load_data('../data/test-news-taskA.jsonl')  # Change to appropriate file for evaluation

    label_choices = list(set(train_labels))  # Assuming binary classification (e.g., [0, 1])

    # Random Baseline
    random_predictions = random_baseline(val_labels, label_choices)
    print("Random Baseline Accuracy:", evaluate_baseline(random_predictions, val_labels))

    # Majority Baseline
    majority_predictions = majority_baseline(train_labels, val_labels)
    print("Majority Baseline Accuracy:", evaluate_baseline(majority_predictions, val_labels))

    # Stratified Baseline
    stratified_predictions = stratified_baseline(train_labels, val_labels)
    print("Stratified Baseline Accuracy:", evaluate_baseline(stratified_predictions, val_labels))
