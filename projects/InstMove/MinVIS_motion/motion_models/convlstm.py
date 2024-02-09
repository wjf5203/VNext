import torch
import torch.nn as nn


class NPUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(NPUnit, self).__init__()
        same_padding = int((kernel_size[0]-1)/2)
        self.conv2d_x = nn.Conv2d(in_channels=in_channels, out_channels=4*out_channels,
                                  kernel_size=kernel_size, stride=1, padding=same_padding, bias=True)
        self.conv2d_h = nn.Conv2d(in_channels=out_channels, out_channels=4*out_channels,
                                  kernel_size=kernel_size, stride=1, padding=same_padding, bias=True)

    def forward(self, x, h, c):
        x_after_conv = self.conv2d_x(x)
        h_after_conv = self.conv2d_h(h)
        xi, xc, xf, xo = torch.chunk(x_after_conv, 4, dim=1)
        hi, hc, hf, ho = torch.chunk(h_after_conv, 4, dim=1)

        it = torch.sigmoid(xi+hi)
        ft = torch.sigmoid(xf+hf)
        new_c = (ft*c)+(it*torch.tanh(xc+hc))
        ot = torch.sigmoid(xo+ho)
        new_h = ot*torch.tanh(new_c)

        return new_h, new_c

class NPUnit_embed(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(NPUnit_embed, self).__init__()
        same_padding = int((kernel_size-1)/2)
        self.conv1d_x = nn.Conv1d(in_channels=in_channels, out_channels=4*out_channels,
                                  kernel_size=kernel_size, stride=1, padding=same_padding, bias=True)
        self.conv1d_h = nn.Conv1d(in_channels=out_channels, out_channels=4*out_channels,
                                  kernel_size=kernel_size, stride=1, padding=same_padding, bias=True)

    def forward(self, x, h, c):
        x_after_conv = self.conv1d_x(x)
        h_after_conv = self.conv1d_h(h)
        xi, xc, xf, xo = torch.chunk(x_after_conv, 4, dim=1)
        hi, hc, hf, ho = torch.chunk(h_after_conv, 4, dim=1)

        it = torch.sigmoid(xi+hi)
        ft = torch.sigmoid(xf+hf)
        new_c = (ft*c)+(it*torch.tanh(xc+hc))
        ot = torch.sigmoid(xo+ho)
        new_h = ot*torch.tanh(new_c)

        return new_h, new_c