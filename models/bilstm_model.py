import torch
import torch.nn as nn


class LSTM_glove_vecs(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, glove_weights, num_labels, dropout_rate):
        super().__init__()
        self.embeddings = nn.Embedding.from_pretrained(torch.from_numpy(glove_weights), freeze=True, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, num_labels)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x, teacher_forcing_ratio=0.5):
        x = self.embeddings(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        emissions = self.linear(lstm_out)

        # Implement teacher forcing
        if self.training and teacher_forcing_ratio > 0.0:
            # Use teacher forcing
            teacher_forcing_mask = torch.rand(x.size(0)) < teacher_forcing_ratio
            modified_input = x.clone()  # Clone the input tensor to avoid inplace modification
            for i in range(x.size(0)):
                if teacher_forcing_mask[i]:
                    modified_input[i, 1:] = self.embeddings(torch.argmax(emissions[i, :-1], dim=-1))
            lstm_out, _ = self.lstm(modified_input)

        return emissions
