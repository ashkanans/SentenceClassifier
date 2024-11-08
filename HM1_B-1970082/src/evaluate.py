import pickle

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

from model import LSTMClassifier
from train import TextDataset, SimpleTokenizer, load_data


# Load the trained model
def load_model(filepath, vocab_size, embedding_dim, hidden_dim, output_dim, padding_idx, bidirectional=True):
    model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, padding_idx, bidirectional=bidirectional)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model


# Evaluation function
def evaluate_model(model, test_loader):
    """Evaluate model performance on the test set."""
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

    # Print metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    return accuracy, precision, recall, f1


# Main evaluation code
if __name__ == "__main__":
    # Model parameters (must match training configuration)
    embedding_dim = 100
    hidden_dim = 128
    output_dim = 2
    max_len = 50

    # Load test data
    test_texts, test_labels = load_data('../data/test-tweets-taskA.jsonl')  # Change as needed for other test sets

    # Load the vocabulary from training for consistent token encoding
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    # Initialize tokenizer with loaded vocabulary
    tokenizer = SimpleTokenizer()
    tokenizer.vocab = vocab  # Use the loaded vocabulary instead of building a new one

    # Prepare DataLoader
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_len)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Load the trained model
    model = load_model(
        'lstm_model.pth',
        vocab_size=len(tokenizer.vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        padding_idx=tokenizer.vocab[tokenizer.pad_token],
        bidirectional=True
    )

    # Evaluate model
    evaluate_model(model, test_loader)
