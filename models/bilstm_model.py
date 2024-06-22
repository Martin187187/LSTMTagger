import torch
import torch.nn as nn


class LSTM_glove_vecs(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, glove_weights, num_labels):
        super().__init__()
        self.embeddings = nn.Embedding.from_pretrained(torch.from_numpy(glove_weights), freeze=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=False, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, num_labels)
        self.dropout = nn.Dropout(0.1)
        # Initialize LSTM weights and biases
        self.init_lstm_weights()

    def forward(self, x):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)  # Apply dropout after LSTM
        logits = self.linear(lstm_out)
        return logits

    def init_lstm_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0.01)  # Initialize biases to 0.01 or as needed
                # Adjust forget gate bias initialization
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)  # Set forget gate biases to 1
