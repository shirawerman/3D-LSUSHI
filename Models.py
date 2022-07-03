from torch import nn #noqa
import torch #noqa

class prox(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.init_parameters()

    def init_parameters(self):
        nn.init.uniform_(self.alpha, b=1e-5)

    def forward(self, x):
        tau = 1
        return x / (1 + torch.exp(-tau * (torch.abs(x) - self.alpha)))


class block(nn.Module):
  """A super resolution model. """

  def __init__(self, kernel_size=[15,5], num_channels=1):
    super().__init__()
    self.prox = prox()
    self.conv1 = nn.Conv3d(in_channels=1, out_channels=num_channels, kernel_size=kernel_size[0], padding=int((kernel_size[0]-1)/2), bias=True)
    self.conv2 = nn.Conv3d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size[1], padding=int((kernel_size[1]-1)/2), bias=True)

  def forward(self, x1, x2):
    y = self.prox(x2)
    y = self.conv1(x1) + self.conv2(y)

    return y


class duulm_net(nn.Module):
  """A super resolution model. """

  def __init__(self, scale_factor, folds=8, kernel_size=[15,5], num_channels=1, interp_mode='nearest'):
    super().__init__()
    self.scale_factor = scale_factor
    self.interp_mode = interp_mode
    self.folds = folds
    self.prox = prox()
    self.conv1 = nn.Conv3d(in_channels=1, out_channels=num_channels, kernel_size=kernel_size[0], padding=int((kernel_size[0]-1)/2), bias=True)
    self.conv2 = nn.Conv3d(in_channels=num_channels, out_channels=1, kernel_size=1, padding=int((1-1)/2), bias=False)
    self.blocks = nn.ModuleList([block(kernel_size=kernel_size, num_channels=num_channels) for i in range(self.folds)])

  def forward(self, x):
    x_scaled = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.interp_mode)

    x_prev = self.conv1(x_scaled)
    for b in self.blocks:
      x_prev = b(x_scaled,x_prev)

    y = self.prox(x_prev)
    y = self.conv2(y)
    return y
  