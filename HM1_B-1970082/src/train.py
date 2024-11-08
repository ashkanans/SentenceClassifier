import json
import pickle

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import LSTMClassifier  # Uses the updated LSTMClassifier


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
    # Hyperparameters and configurations
    embedding_dim = 100
    hidden_dim = 128
    output_dim = 2  # binary classification
    batch_size = 32
    num_epochs = 10
    max_len = 50
    learning_rate = 0.001

    # Load training data
    train_texts, train_labels = load_data('../data/train-taskA.jsonl')

    # Initialize tokenizer and build vocabulary
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(train_texts)

    # Save the vocabulary for consistent encoding in evaluation
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(tokenizer.vocab, f)

    # Prepare DataLoader
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model with the new architecture
    model = LSTMClassifier(
        vocab_size=len(tokenizer.vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        padding_idx=tokenizer.vocab[tokenizer.pad_token],
        bidirectional=True
    )

    # Use CrossEntropyLoss and Adam optimizer with weight decay for regularization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

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

            # Accumulate loss and accuracy
            epoch_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()

        # Calculate epoch accuracy
        accuracy = correct_predictions / len(train_dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'lstm_model.pth')


# Run training
if __name__ == "__main__":
    train_model()
