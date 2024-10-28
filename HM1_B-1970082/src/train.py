import json
import pickle

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import LSTMClassifier


# Simple Tokenizer with UNK handling
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"

    def build_vocab(self, texts):
        words = set(word for text in texts for word in text.split())
        self.vocab = {word: idx + 2 for idx, word in enumerate(words)}  # +2 to reserve 0 for PAD and 1 for UNK
        self.vocab[self.pad_token] = 0
        self.vocab[self.unk_token] = 1

    def encode(self, text, max_len):
        tokens = [self.vocab.get(word, self.vocab[self.unk_token]) for word in text.split()]
        return tokens[:max_len] + [self.vocab[self.pad_token]] * (max_len - len(tokens))


# Dataset class for loading data
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.tokenizer.encode(self.texts[idx], self.max_len))
        label = torch.tensor(self.labels[idx])
        return input_ids, label


# Load data
def load_data(filepath):
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]
    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]
    return texts, labels


# Main training function
def train_model():
    # Parameters
    embedding_dim = 100
    hidden_dim = 128
    output_dim = 2  # binary classification
    batch_size = 32
    num_epochs = 10
    max_len = 50

    # Load data
    train_texts, train_labels = load_data('../data/train-taskA.jsonl')

    # Initialize simple tokenizer and build vocab from training data only
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(train_texts)

    # Save the vocabulary to a file for evaluation use
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(tokenizer.vocab, f)

    # Prepare DataLoader
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = LSTMClassifier(len(tokenizer.vocab), embedding_dim, hidden_dim, output_dim,
                           padding_idx=tokenizer.vocab[tokenizer.pad_token], bidirectional=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct_predictions = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()

        accuracy = correct_predictions / len(train_dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader)}, Accuracy: {accuracy:.4f}')

    # Save the model after training
    torch.save(model.state_dict(), 'lstm_model.pth')


# Run training
if __name__ == "__main__":
    train_model()
