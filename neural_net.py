import torch
import torch.nn as nn


class NeuralColumn(nn.Module):
    def __init__(self, channels: int, output_dim: int) -> None:
        """ channels is the number of output convolution channels for
            each convolution layer of the network, except the last one.
        """
        super(NeuralColumn, self).__init__()
        self._conv_net = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=5),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(channels, output_dim, kernel_size=3, bias=False)
        )

    def output_dim(self) -> int:
        return self._conv_net[-1].weight.size(0)

    def channels(self) -> int:
        return self._conv_net[0].weight.size(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conved = self._conv_net(x)

        # global pooling on x and y axis
        pooled = conved.max(dim=3)[0].max(dim=2)[0]

        return pooled.view(x.size(0), self.output_dim())


class Ensemble(nn.Module):
    def __init__(self, n_columns: int, column_dim: int, channels: int) -> None:
        super(Ensemble, self).__init__()
        self.columns = nn.ModuleList([
            NeuralColumn(channels, column_dim) for _ in range(n_columns)
        ])

    def num_columns(self) -> int:
        return len(self.columns)

    def channels(self) -> int:
        return self.columns[0].channels()

    def column_dim(self) -> int:
        return self.columns[0].output_dim()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            column(x) for column in self.columns
        ], dim=1)


def load_model(path: str) -> nn.Module:
    print("loading model")
    parts = path.split("_")

    columns, channels, col_dim = parts[-3:]
    columns = int(columns)
    channels = int(channels)
    col_dim = int(col_dim)

    net = Ensemble(n_columns=columns, column_dim=col_dim, channels=channels)
    net.load_state_dict(torch.load(path))
    net.eval()

    for param in net.parameters():
        param.requires_grad_(False)

    print("model loaded")

    return net
