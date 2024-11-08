import gc
import json
import os
import pickle
from multiprocessing import Pool, cpu_count

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import LSTMClassifier  # Ensure this is the updated LSTM model

# Directory for storing models
MODELS_DIR = 'trained_models'
best_model_info = {
    "name": "",
    "path": "",
    "hyperparams": {},
    "validation_accuracy": 0
}


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


# Load data from JSONL format
def load_data(filepath):
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]
    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]
    return texts, labels


def save_hyperparams(params, save_path):
    """Save hyperparameters to a JSON file in the model's directory."""
    with open(os.path.join(save_path, 'hyperparams.json'), 'w') as f:
        json.dump(params, f, indent=4)


def save_training_metrics(metrics, save_path):
    """Save training and validation metrics to a JSON file."""
    with open(os.path.join(save_path, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)


def train_model(embedding_dim=100, hidden_dim=128, output_dim=2, batch_size=32, num_epochs=10, max_len=50,
                learning_rate=0.001, weight_decay=1e-5, patience=3, model_save_path='best_lstm_model.pth'):
    # Load and split data
    train_texts, train_labels = load_data('../data/train-taskA.jsonl')
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.1)

    # Initialize tokenizer and vocabulary
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(train_texts)
    with open(os.path.join(model_save_path, 'vocab.pkl'), 'wb') as f:
        pickle.dump(tokenizer.vocab, f)

    # Prepare DataLoaders for training and validation
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize the LSTM model
    model = LSTMClassifier(
        vocab_size=len(tokenizer.vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        padding_idx=tokenizer.vocab[tokenizer.pad_token],
        bidirectional=True
    )

    # Define loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)

    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0

    # Track metrics for each epoch
    metrics = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": [], "epoch_details": []}

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct_predictions = 0, 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = correct_predictions / len(train_dataset)

        # Store detailed metrics for this epoch
        epoch_detail = f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}'
        metrics["train_loss"].append(train_loss)
        metrics["train_accuracy"].append(train_accuracy)

        model.eval()
        val_loss, val_correct = 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                val_correct += (predictions == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = val_correct / len(val_dataset)

        # Store validation metrics for this epoch
        metrics["val_loss"].append(val_loss)
        metrics["val_accuracy"].append(val_accuracy)

        # Log the epoch results
        epoch_detail += f'\nValidation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}'
        metrics["epoch_details"].append(epoch_detail)

        print(epoch_detail)

        # Check for best model based on validation accuracy
        if val_accuracy > best_model_info["validation_accuracy"]:
            best_model_info.update({
                "name": f"emb{embedding_dim}_hid{hidden_dim}_ep{num_epochs}_len{max_len}_bs{batch_size}_lr{learning_rate}_wd{weight_decay}",
                "path": model_save_path,
                "hyperparams": {
                    "embedding_dim": embedding_dim,
                    "hidden_dim": hidden_dim,
                    "num_epochs": num_epochs,
                    "max_len": max_len,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay
                },
                "validation_accuracy": val_accuracy
            })

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_save_path, 'model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Save metrics
    save_training_metrics(metrics, model_save_path)

    # Clear memory after training
    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()


def process_training(params):
    embedding_dim, hidden_dim, num_epochs, max_len, batch_size, learning_rate, weight_decay = params

    # Create a unique directory for each model
    model_dir = os.path.join(MODELS_DIR, f'emb{embedding_dim}_hid{hidden_dim}_ep{num_epochs}_len{max_len}_'
                                         f'bs{batch_size}_lr{learning_rate}_wd{weight_decay}')
    os.makedirs(model_dir, exist_ok=True)

    # Save hyperparameters
    save_hyperparams({
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'num_epochs': num_epochs,
        'max_len': max_len,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay
    }, model_dir)

    # Train the model
    train_model(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=2,
        batch_size=batch_size,
        num_epochs=num_epochs,
        max_len=max_len,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        model_save_path=model_dir
    )


if __name__ == "__main__":
    # Hyperparameter values to test
    hyperparameter_values = {
        'embedding_dim': [50, 100, 200],
        'hidden_dim': [64, 128, 256],
        'num_epochs': [5, 10],
        'max_len': [30, 50],
        'batch_size': [16, 32],
        'learning_rate': [0.0001, 0.001],
        'weight_decay': [0, 1e-5]
    }

    # Prepare combinations for multiprocessing
    params_list = [
        (embedding_dim, hidden_dim, num_epochs, max_len, batch_size, learning_rate, weight_decay)
        for embedding_dim in hyperparameter_values['embedding_dim']
        for hidden_dim in hyperparameter_values['hidden_dim']
        for num_epochs in hyperparameter_values['num_epochs']
        for max_len in hyperparameter_values['max_len']
        for batch_size in hyperparameter_values['batch_size']
        for learning_rate in hyperparameter_values['learning_rate']
        for weight_decay in hyperparameter_values['weight_decay']
    ]

    # Limit to half available CPUs
    with Pool(cpu_count() // 2) as pool:
        pool.map(process_training, params_list)

    # Save the best model info after training completes
    with open(os.path.join(MODELS_DIR, 'best_model_info.json'), 'w') as f:
        json.dump(best_model_info, f, indent=4)
