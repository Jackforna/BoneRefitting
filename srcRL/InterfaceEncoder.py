import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import SAGEConv, global_max_pool, global_mean_pool

# Layout delle feature per nodo dell'osservazione "iface" prodotta da
# ArchaeologicalIncastroEnv._build_interface_features. Se cambi il
# layout lato ambiente, aggiorna ANCHE queste costanti: gli indici
# sono usati per costruire il grafo (posizioni e lato A/B).
IFACE_POS_COLS = slice(0, 3)   # posizione relativa al centro frattura di A
IFACE_SIDE_COL = 6             # 0 = frammento A, 1 = frammento B
IFACE_NUM_FEATURES = 13


class InterfaceGraphEncoder(nn.Module):
    """
    GNN sul grafo congiunto dell'interfaccia di frattura A-B.

    I nodi sono punti delle maschere di frattura di A e di B nella
    posa CORRENTE. Gli archi sono:
      - kNN intra-frammento (struttura locale della superficie);
      - kNN inter-frammento (A->B e B->A): collegano ogni punto ai
        punti piu' vicini dell'altro osso, quindi il message passing
        "vede" come le due superfici si stanno toccando (gap,
        compenetrazioni, disallineamenti locali).

    Il grafo viene ricostruito ad ogni forward dalle posizioni, quindi
    cambia ad ogni step dell'agente. Usa LayerNorm invece di BatchNorm:
    in PPO la raccolta dei rollout avviene con batch da 1 e in modalita'
    eval, dove BatchNorm si comporta in modo diverso dal training.
    """

    def __init__(self, in_channels=IFACE_NUM_FEATURES, hidden_channels=64, out_channels=128,
                 k_intra=12, k_cross=4):
        super().__init__()
        self.k_intra = k_intra
        self.k_cross = k_cross

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)

        # mean + max, separati per lato A e lato B
        self.out_dim = out_channels * 4

    @torch.no_grad()
    def _build_edge_index(self, pos, side):
        """
        pos: (B, N, 3), side: (B, N) con valori 0/1.
        Restituisce edge_index (2, E) sul batch appiattito (B*N nodi),
        in convenzione PyG source -> target.
        """
        b, n, _ = pos.shape
        dist = torch.cdist(pos, pos)
        same_side = side.unsqueeze(2) == side.unsqueeze(1)
        eye = torch.eye(n, dtype=torch.bool, device=pos.device).unsqueeze(0)

        inf = torch.finfo(dist.dtype).max
        d_intra = dist.masked_fill(~same_side | eye, inf)
        d_cross = dist.masked_fill(same_side, inf)

        # Numero di vicini limitato dai punti disponibili per lato
        n_a = int((side[0] == 0).sum().item())
        n_b = n - n_a
        k_intra = max(1, min(self.k_intra, min(n_a, n_b) - 1))
        k_cross = max(1, min(self.k_cross, min(n_a, n_b)))

        nbr_intra = torch.topk(d_intra, k_intra, dim=-1, largest=False).indices
        nbr_cross = torch.topk(d_cross, k_cross, dim=-1, largest=False).indices
        nbr = torch.cat([nbr_intra, nbr_cross], dim=-1)  # (B, N, K)

        offset = (torch.arange(b, device=pos.device) * n).view(b, 1, 1)
        target = (torch.arange(n, device=pos.device).view(1, n, 1) + offset).expand_as(nbr)
        source = nbr + offset
        return torch.stack([source.reshape(-1), target.reshape(-1)], dim=0)

    def forward(self, x):
        """x: (B, N, F) feature per nodo. Restituisce (B, out_dim)."""
        b, n, f = x.shape
        pos = x[..., IFACE_POS_COLS]
        side = (x[..., IFACE_SIDE_COL] > 0.5).long()

        edge_index = self._build_edge_index(pos, side)
        h = x.reshape(b * n, f)

        h = F.relu(self.norm1(self.conv1(h, edge_index)))
        h = F.relu(self.norm2(self.conv2(h, edge_index)))
        h = self.conv3(h, edge_index)

        # Pooling separato per lato: batch id "grafo" = 2*sample + lato
        batch = (torch.arange(b, device=x.device).repeat_interleave(n) * 2) + side.reshape(-1)
        pooled_mean = global_mean_pool(h, batch, size=2 * b).view(b, -1)
        pooled_max = global_max_pool(h, batch, size=2 * b).view(b, -1)
        return torch.cat([pooled_mean, pooled_max], dim=-1)


class InterfaceFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor per stable-baselines3 (MultiInputPolicy).

    Riceve l'osservazione Dict dell'ambiente:
      - "geom":  vettore delle metriche geometriche (chamfer, normali, ...)
      - "iface": (N, F) nodi del grafo dell'interfaccia A-B
    e restituisce la concatenazione [embedding GNN, geom]. La GNN viene
    addestrata end-to-end dai gradienti di PPO (actor e critic).
    """

    def __init__(self, observation_space: gym.spaces.Dict, hidden_channels=64, out_channels=128,
                 k_intra=12, k_cross=4):
        iface_space = observation_space["iface"]
        geom_dim = observation_space["geom"].shape[0]

        encoder = InterfaceGraphEncoder(
            in_channels=iface_space.shape[1],
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            k_intra=k_intra,
            k_cross=k_cross,
        )
        super().__init__(observation_space, features_dim=encoder.out_dim + geom_dim)
        self.encoder = encoder

    def forward(self, observations):
        emb = self.encoder(observations["iface"])
        return torch.cat([emb, observations["geom"]], dim=-1)