
# Import modules
import torch
import torch.nn as nn


class Lstm(nn.Module):
    def __init__(self, input_feat, pred_len, hidden_units, num_layers):
        super().__init__()
        self.input_feat = input_feat
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.pred_len = pred_len

        self.lstm = nn.LSTM(
            input_size=self.input_feat, hidden_size=self.hidden_units,
            num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(
            in_features=hidden_units,
            out_features=self.pred_len)

    def forward(self, input):
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_units)
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_units)

        # input -> (B, input_len, 1)
        x, _ = self.lstm(input, (h0, c0))
        # x -> (B, input_len, hidden_units)
        out = self.linear(x[:, -1, :])
        # out -> (B, pred_len, 1)
        out = torch.reshape(out, (out.size(0), self.pred_len, self.input_feat))

        return out
