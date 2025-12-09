# recurrent module (representation unit)
# convoLSTM2D -produces-> Rl

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

# updates Rl
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2 # to ensure that spatial size stays constant (same HxW out as in)

        # all gates (i, f, o, g) in one conv computed
        # input: Al, El, Rl+1, hprev
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding
        )

        self.hidden_channels = hidden_channels # number of neurons in Rl

    def forward(self, x, hidden):
        """
        x = input tensor (A_l concat E_l concat R_{l+1})
        hidden = (h, c)
        """
        h_prev, c_prev = hidden # unpacks the previous hidden state
        combined = torch.cat([x, h_prev], dim=1) # concetanates input and memory along channel dimension to update Rl

        gates = self.conv(combined) # runs 1 convolution to compute gates
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1) # splits the 4 gate tensors

        # applies non-linearity to each gate, to prevnet exploding gradients
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        # (standard LSTM template)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, c
