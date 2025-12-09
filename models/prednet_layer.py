# models/prednet_layer.py

# single layer of prednet
import torch
import torch.nn as nn
import torch.nn.functional as F
from .convlstm import ConvLSTMCell

class PredNetLayer(nn.Module):
    def __init__(self, A_channels, R_channels, Ahat_channels):
        super().__init__()

        # R update (ConvLSTM)
        self.convlstm = ConvLSTMCell(
            in_channels=A_channels + 2*A_channels + R_channels,   # A_l + E_l + R_up
            hidden_channels=R_channels
        )

        # Ahat generator
        self.ahat = nn.Conv2d(R_channels, Ahat_channels, kernel_size=3, padding=1)

        # A_l (next layer input)
        self.A = nn.Conv2d(2 * Ahat_channels, A_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, A_l, E_l, R_up, hidden_state):
        # Update R
        lstm_input = torch.cat([A_l, E_l, R_up], dim=1)
        R_l, C_l = self.convlstm(lstm_input, hidden_state)

        # Predict Ahat
        Ahat_l = F.relu(self.ahat(R_l))

        # Compute error
        E_plus  = F.relu(A_l - Ahat_l)
        E_minus = F.relu(Ahat_l - A_l)
        E_l_new = torch.cat([E_plus, E_minus], dim=1)

        # Bottom-up A for next layer
        A_next = self.A(E_l_new)

        return R_l, C_l, Ahat_l, E_l_new, A_next
