import torch
import torch.nn as nn


class LSTM_glove_vecs(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, glove_weights, num_labels, dropout_rate):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding.from_pretrained(torch.from_numpy(glove_weights), freeze=True, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, num_labels)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x, teacher_forcing_ratio=0.0):
        x = self.embeddings(x)
        batch_size = x.size(0)
        h_0 = torch.zeros(2, batch_size, self.hidden_dim).to(x.device)  # 2 for bidirection
        c_0 = torch.zeros(2, batch_size, self.hidden_dim).to(x.device)

        lstm_out, _ = self.lstm(x, (h_0, c_0))
        lstm_out = self.dropout(lstm_out)
        emissions = self.linear(lstm_out)

        # Implement teacher forcing
        if self.training and teacher_forcing_ratio > 0.0:
            # Use teacher forcing
            teacher_forcing_mask = torch.rand(x.size(0)) < teacher_forcing_ratio
            modified_input = x.clone()
            for i in range(x.size(0)):
                if teacher_forcing_mask[i]:
                    modified_input[i, 1:] = self.embeddings(torch.argmax(emissions[i, :-1], dim=-1))
            lstm_out, _ = self.lstm(modified_input)

        return emissions
