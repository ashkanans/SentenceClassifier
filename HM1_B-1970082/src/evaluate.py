import pickle

import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from model import LSTMClassifier
from train import TextDataset, SimpleTokenizer, load_data


# Load the trained model
def load_model(filepath, vocab_size, embedding_dim, hidden_dim, output_dim, padding_idx):
    model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, padding_idx, bidirectional=True)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model


# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    all_predictions, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy


# Main evaluation code
if __name__ == "__main__":
    # Parameters
    embedding_dim = 100
    hidden_dim = 128
    output_dim = 2
    max_len = 50

    # Load test data
    test_texts, test_labels = load_data('../data/test-news-taskA.jsonl')  # Change for other test sets as needed

    # Load the saved vocabulary for consistent encoding
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    # Initialize tokenizer and set vocabulary to loaded vocab
    tokenizer = SimpleTokenizer()
    tokenizer.vocab = vocab  # Use the loaded vocabulary instead of building a new one

    # Prepare DataLoader
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_len)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Load the trained model
    model = load_model('lstm_model.pth', len(tokenizer.vocab), embedding_dim, hidden_dim, output_dim,
                       padding_idx=tokenizer.vocab[tokenizer.pad_token])

    # Evaluate
    evaluate_model(model, test_loader)
