import json
import random
from collections import Counter

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# Load and preprocess data
def load_data(filepath):
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]
    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]
    return texts, labels


# Baseline functions
def random_baseline(test_data, label_choices):
    """Predicts random labels chosen uniformly from the label set."""
    return [random.choice(label_choices) for _ in test_data]


def majority_baseline(train_labels, test_data):
    """Predicts the most common label from the training data."""
    majority_label = Counter(train_labels).most_common(1)[0][0]
    return [majority_label] * len(test_data)


def stratified_baseline(train_labels, test_data):
    """Predicts labels according to the distribution in the training data."""
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
    """Calculates accuracy, precision, recall, and F1-score for the predictions."""
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    return accuracy, precision, recall, f1


# Main execution
if __name__ == "__main__":
    # Load training and validation data
    train_texts, train_labels = load_data('../data/train-taskA.jsonl')
    _, val_labels = load_data('../data/test-news-taskA.jsonl')  # Change to appropriate file for evaluation

    label_choices = list(set(train_labels))  # Assuming binary classification (e.g., [0, 1])

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
