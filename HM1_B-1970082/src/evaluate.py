import json
import os
import pickle

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

from model import LSTMClassifier
from train import TextDataset, SimpleTokenizer, load_data


# Load the trained model with specified hyperparameters
def load_model(filepath, vocab_size, embedding_dim, hidden_dim, output_dim, padding_idx, bidirectional=True):
    model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, padding_idx, bidirectional=bidirectional)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model


# Evaluation function for a single model on a given test set
def evaluate_model(model, test_loader):
    """Evaluate model performance on a test set."""
    model.eval()
    all_predictions, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return accuracy, precision, recall, f1


# Main evaluation function to loop through all models in directory and test on multiple datasets
def main_evaluation(models_dir, test_sets):
    best_model_info = {"name": "", "path": "", "test_set": "", "metrics": {}}
    highest_f1 = 0  # Track highest F1-score across all models and test sets

    # Loop through all models in the models directory
    for model_name in os.listdir(models_dir):
        model_dir = os.path.join(models_dir, model_name)
        model_path = os.path.join(model_dir, 'model.pth')
        vocab_path = os.path.join(model_dir, 'vocab.pkl')
        hyperparams_path = os.path.join(model_dir, 'hyperparams.json')

        if not os.path.isfile(model_path) or not os.path.isfile(vocab_path) or not os.path.isfile(hyperparams_path):
            continue

        # Load model hyperparameters
        with open(hyperparams_path, 'r') as f:
            hyperparams = json.load(f)

        # Load vocabulary and initialize tokenizer
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        tokenizer = SimpleTokenizer()
        tokenizer.vocab = vocab

        # Load model with hyperparameters
        model = load_model(
            filepath=model_path,
            vocab_size=len(vocab),
            embedding_dim=hyperparams['embedding_dim'],
            hidden_dim=hyperparams['hidden_dim'],
            output_dim=2,  # Adjust as needed for multi-class classification
            padding_idx=tokenizer.vocab[tokenizer.pad_token],
            bidirectional=hyperparams['bidirectional']
        )

        # Evaluate on each test set
        for test_name, test_filepath in test_sets.items():
            # Load test data and prepare DataLoader
            test_texts, test_labels = load_data(test_filepath)
            test_dataset = TextDataset(test_texts, test_labels, tokenizer, hyperparams['max_len'])
            test_loader = DataLoader(test_dataset, batch_size=32)

            # Evaluate model and capture metrics
            accuracy, precision, recall, f1 = evaluate_model(model, test_loader)
            print(
                f"Model: {model_name}, Test Set: {test_name}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

            # Update best model info based on F1 score
            if f1 > highest_f1:
                highest_f1 = f1
                best_model_info = {
                    "name": model_name,
                    "path": model_path,
                    "test_set": test_name,
                    "metrics": {
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1
                    }
                }

    # Print the best model and its performance
    print("\nBest Model Information:")
    print(f"Model Name: {best_model_info['name']}")
    print(f"Path: {best_model_info['path']}")
    print(f"Test Set: {best_model_info['test_set']}")
    print("Metrics:")
    for metric, value in best_model_info['metrics'].items():
        print(f"{metric.capitalize()}: {value:.4f}")


if __name__ == "__main__":
    # Directory containing the trained models
    models_dir = 'trained_models'

    # Define the test sets
    test_sets = {
        "test_tweets": '../data/test-tweets-taskA.jsonl',
        "test_news": '../data/test-news-taskA.jsonl'
    }

    # Perform evaluation on all models and find the best one
    main_evaluation(models_dir, test_sets)
