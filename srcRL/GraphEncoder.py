import torch
import torch.nn.functional as F

from torch_geometric.nn import (
    SAGEConv,
    global_mean_pool,
    global_max_pool,
    BatchNorm
)

class GraphEncoder(
    torch.nn.Module
):

    def __init__(self, in_channels=6, hidden_channels=64, out_channels=128):

        super().__init__()

        self.conv1 = SAGEConv(in_channels,hidden_channels)
        self.conv2 = SAGEConv(64,128)
        self.conv3 = SAGEConv(128,256)

        self.bn1 = BatchNorm(64)
        self.bn2 = BatchNorm(128)

    def forward(self,data):

        x = self.conv1(data.x,data.edge_index)

        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x,0.2,training=self.training)


        x = self.conv2(x,data.edge_index)

        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x,0.2,training=self.training)


        x = self.conv3(x,data.edge_index)

        batch = getattr(data,"batch",None)

        if batch is None:

            batch = torch.zeros(
                len(x),
                dtype=torch.long,
                device=x.device
            )

        # 🚀 DUAL POOLING: Mean (Struttura) + Max (Dettagli acuti della frattura)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)

        # Concateniamo per ottenere un vettore rappresentativo ricco (dimensione: out_channels * 2)
        out = torch.cat([x_mean, x_max], dim=-1)

        return x