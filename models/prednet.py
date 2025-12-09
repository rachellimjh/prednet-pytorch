# models/prednet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .prednet_layer import PredNetLayer

class PredNet(nn.Module):
    def __init__(self, layer_channels):
        """
        layer_channels = [
            (A0, R0),
            (A1, R1),
            (A2, R2),
            ...
        ]
        """
        super().__init__()
        self.num_layers = len(layer_channels)

        self.layers = nn.ModuleList()
        for i, (A_ch, R_ch) in enumerate(layer_channels):
            Ahat_ch = A_ch
            self.layers.append(PredNetLayer(A_ch, R_ch, Ahat_ch))

    def init_states(self, batch, size_hw):
        states = []
        for (A_ch, R_ch) in [(A, R) for A, R in layer_channels]:
            H, W = size_hw
            R_init = torch.zeros(batch, R_ch, H, W)
            C_init = torch.zeros(batch, R_ch, H, W)
            states.append([R_init, C_init])
            H //= 2
            W //= 2
        return states

    def forward(self, input_sequence):
        """
        input_sequence: (B, T, C, H, W)
        """
        B, T, C, H, W = input_sequence.shape

        # Init states
        layer_states = [
            (torch.zeros(B, R, H // (2**l), W // (2**l)), 
             torch.zeros(B, R, H // (2**l), W // (2**l)))
            for l, (_, R) in enumerate(layer_channels)
        ]

        outputs = []

        for t in range(T):
            A = input_sequence[:, t]  # A_0
            E = torch.zeros(B, C*2, H, W)  # initial E_0
            R_up = torch.zeros_like(layer_states[-1][0])  # no R_up for top layer

            # Top-down update for all layers
            for l in reversed(range(self.num_layers)):
                R, C = layer_states[l]
                R_new, C_new, Ahat, E_new, A_next = self.layers[l](A, E, R_up, (R, C))
                layer_states[l] = (R_new, C_new)

                R_up = R_new  # pass down
                A = A_next    # for next layer
                E = E_new     # for next layer

            outputs.append(Ahat)

        return torch.stack(outputs, dim=1)  # (B, T, C, H, W)
