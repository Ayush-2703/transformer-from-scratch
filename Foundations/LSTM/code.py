"""
LSTM implementation from scratch in PyTorch.
Shows all four gates explicitly.
"""

import torch
import torch.nn as nn


class LSTMCellManual(nn.Module):
    """
    Single LSTM time step — all gates written out explicitly.
    Inputs:
        x_t:  (batch, input_size)
        h_{t-1}: (batch, hidden_size)
        c_{t-1}: (batch, hidden_size)
    Outputs:
        h_t, c_t
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Each gate: input → hidden  +  hidden → hidden
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)  # forget
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)  # input
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)  # candidate
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)  # output

    def forward(self, x_t, h_prev, c_prev):
        combined = torch.cat([h_prev, x_t], dim=-1)  # [h_{t-1}, x_t]

        f_t = torch.sigmoid(self.W_f(combined))          # forget gate
        i_t = torch.sigmoid(self.W_i(combined))          # input gate
        C_tilde = torch.tanh(self.W_c(combined))         # candidate memory
        o_t = torch.sigmoid(self.W_o(combined))          # output gate

        c_t = f_t * c_prev + i_t * C_tilde               # new cell state
        h_t = o_t * torch.tanh(c_t)                      # new hidden state

        return h_t, c_t


class LSTMSequence(nn.Module):
    """
    Runs LSTMCell over a full sequence.
    Input:  (batch, seq_len, input_size)
    Output: (batch, seq_len, hidden_size), (h_n, c_n)
    """
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            LSTMCellManual(
                input_size if i == 0 else hidden_size,
                hidden_size
            )
            for i in range(num_layers)
        ])

    def forward(self, x, init_states=None):
        B, T, _ = x.shape

        if init_states is None:
            h = [torch.zeros(B, self.hidden_size, device=x.device)
                 for _ in range(self.num_layers)]
            c = [torch.zeros(B, self.hidden_size, device=x.device)
                 for _ in range(self.num_layers)]
        else:
            h, c = init_states

        outputs = []
        for t in range(T):
            layer_input = x[:, t, :]
            for layer_idx in range(self.num_layers):
                h[layer_idx], c[layer_idx] = self.cells[layer_idx](
                    layer_input, h[layer_idx], c[layer_idx]
                )
                layer_input = h[layer_idx]
            outputs.append(h[-1].unsqueeze(1))

        return torch.cat(outputs, dim=1), (h, c)


# ─── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    batch, seq_len, input_size, hidden_size = 4, 10, 16, 32

    model = LSTMSequence(input_size, hidden_size, num_layers=2)
    x = torch.randn(batch, seq_len, input_size)

    out, (h_n, c_n) = model(x)

    print(f"Input:   {x.shape}")
    print(f"Output:  {out.shape}")          # (4, 10, 32)
    print(f"h_n[-1]: {h_n[-1].shape}")     # (4, 32)  — last layer, last step
