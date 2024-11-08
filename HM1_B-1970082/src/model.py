import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout, num_layers=1, padding_idx=0,
                 bidirectional=False, dropout_rate=0.5):
        """
        Initialize the LSTM-based text classifier model.
        Args:
            vocab_size: Size of the vocabulary.
            embedding_dim: Dimensionality of the embeddings.
            hidden_dim: Number of units in the LSTM's hidden layer.
            output_dim: Number of output classes (e.g., 2 for binary classification).
            num_layers: Number of stacked LSTM layers.
            padding_idx: Index for the padding token to be ignored in embedding.
            bidirectional: Whether the LSTM should be bidirectional.
            dropout_rate: Dropout rate to be applied to the layers.
        """
        super(LSTMClassifier, self).__init__()

        # Embedding layer with padding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        # LSTM layer with specified number of layers and dropout for regularization
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,  # Number of stacked LSTM layers
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout  # Dropout only applies if more than one layer
        )

        # Compute the output dimension based on bidirectionality
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Fully connected layer for classification
        self.fc = nn.Linear(lstm_output_dim, output_dim)

        # Dropout for the embedding layer and hidden layer output
        self.dropout = nn.Dropout(dropout_rate)

        # Initialization of weights
        self._init_weights()

    def _init_weights(self):
        """Applies weight initialization to improve training stability."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)  # Xavier initialization for weights
            elif 'bias' in name:
                nn.init.zeros_(param)  # Zero initialization for biases

    def forward(self, x):
        """
        Forward pass for the LSTM model.
        Args:
            x: Input tensor with token indices.
        Returns:
            Logits for each class in the classification task.
        """
        # Apply embedding layer with dropout
        embedded = self.dropout(self.embedding(x))

        # LSTM layer output
        lstm_out, (hidden, _) = self.lstm(embedded)

        # Concatenate hidden states for bidirectional LSTM
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]),
                               dim=1)  # Concatenate final forward and backward hidden states
        else:
            hidden = hidden[-1]  # Use the last hidden state for the unidirectional case

        # Apply fully connected layer to the hidden state
        logits = self.fc(hidden)

        return logits
