import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class BoneFragmentsDataset(Dataset):
    """Dataset che carica le nuvole di punti/grafi dal disco"""
    def __init__(self, processed_dir="data/processed", num_samples=None):
        self.processed_dir = processed_dir
        
        all_files = sorted([f for f in os.listdir(processed_dir) if f.endswith('.pt')])
        
        self.pairs = []
        for i in range(0, len(all_files), 2):
            if i + 1 < len(all_files):
                self.pairs.append((all_files[i], all_files[i+1]))

        if num_samples is not None:
            self.pairs = self.pairs[:num_samples]

    def __len__(self):
        return len(self.pairs)

    def _build_knn_edge_index(self, pos, k=16):
        """Costruisce edge_index (KNN) in PyTorch puro senza dipendenze C++"""
        with torch.no_grad():
            dist = torch.cdist(pos, pos)
            k_actual = min(k + 1, pos.size(0))
            _, topk_indices = torch.topk(dist, k=k_actual, largest=False, dim=1)
            neighbors = topk_indices[:, 1:]  # Scarta il punto stesso (distanza 0)
            
            num_nodes = pos.size(0)
            row = torch.arange(num_nodes, device=pos.device).repeat_interleave(neighbors.size(1))
            col = neighbors.flatten()
            
            return torch.stack([row, col], dim=0)

    def _to_pyg_data(self, item):
        """Converte un Tensor in oggetto PyG Data"""
        if isinstance(item, Data):
            return item
        
        pos = item[:, :3]
        edge_index = self._build_knn_edge_index(pos, k=16)
        
        return Data(x=item, pos=pos, edge_index=edge_index)

    def __getitem__(self, idx):
        file_a, file_b = self.pairs[idx]
        
        raw_a = torch.load(os.path.join(self.processed_dir, file_a), map_location='cpu')
        raw_b = torch.load(os.path.join(self.processed_dir, file_b), map_location='cpu')
        
        # Convertiamo i tensori grezzi in grafi PyG pronti per il GraphEncoder
        graph_a = self._to_pyg_data(raw_a)
        graph_b = self._to_pyg_data(raw_b)
        
        # 🚀 FIX: Restituiamo i tensori fisici per l'ambiente + gli oggetti Data per l'encoder
        return raw_a, raw_b, (graph_a, graph_b)