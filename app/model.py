import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, drop_prob, num_layers=1, bidir=False, seq="lstm"):
        super(RNNModel, self).__init__()
        self.seq = seq
        self.bidir_f = 2 if bidir else 1
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if seq == "lstm":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidir)
        else:
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidir)

        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim * self.bidir_f, num_classes)

    def forward(self, text_indices):
        embedded_text = self.embedding(text_indices)
        rnn_output, _ = self.rnn(embedded_text)
        # Берем последний скрытый слой
        last_rnn_output = rnn_output[:, -1, :]
        x = self.dropout(last_rnn_output)
        return self.fc(x)