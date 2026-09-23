import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super(EdgeConvLayer, self).__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        b, n, c = x.shape
        dist = torch.cdist(x[..., :3], x[..., :3])
        _, nn_idx = torch.topk(dist, k=self.k, dim=-1, largest=False)
        
        batch_idx = torch.arange(b).view(-1, 1, 1).expand(b, n, self.k)
        x_neighbors = x[batch_idx, nn_idx, :]
        x_central = x.unsqueeze(2).expand(b, n, self.k, c)
        edge_features = torch.cat([x_central, x_neighbors - x_central], dim=-1)
        
        edge_features = edge_features.view(-1, 2 * c)
        out = self.mlp(edge_features)
        out = out.view(b, n, self.k, -1)
        out = torch.max(out, dim=2)[0]
        return out

class SVDBoneRegistration(nn.Module):
    def __init__(self, k=20, feature_dim=128):
        super(SVDBoneRegistration, self).__init__()
        self.k = k
        self.gnn1 = EdgeConvLayer(in_channels=9, out_channels=64, k=self.k)
        self.gnn2 = EdgeConvLayer(in_channels=64, out_channels=feature_dim, k=self.k)
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def _extract_local_features(self, frag):
        feat = self.gnn1(frag)
        feat = self.gnn2(feat)
        feat = self.proj(feat)
        return F.normalize(feat, p=2, dim=-1)

    def forward(self, frag_a, frag_b):
        b, n, _ = frag_a.shape
        _, m, _ = frag_b.shape
        
        xyz_a = frag_a[..., :3] 
        xyz_b = frag_b[..., :3] 

        feat_a = self._extract_local_features(frag_a) 
        feat_b = self._extract_local_features(frag_b) 

        #similarity_matrix = torch.bmm(feat_a, feat_b.transpose(1, 2))
        #P = F.softmax(similarity_matrix, dim=-1)

        similarity_matrix = torch.bmm(feat_a, feat_b.transpose(1, 2))

        # Estraiamo le feature d'asse Z che abbiamo iniettato (canale 6)
        # frag_a e frag_b hanno dimensioni [B, N, 9]
        z_a = frag_a[..., 6].unsqueeze(-1)  # [B, N, 1]
        z_b = frag_b[..., 6].unsqueeze(1)   # [B, 1, M]

        # Calcoliamo la distanza tra i tag di posizione dei punti.
        # Se un punto è all'estremità (Z vicino a 1) e l'altro è al centro (Z vicino a 0.5), 
        # la distanza sarà circa 0.5.
        distanza_asse = torch.abs(z_a - z_b) # [B, N, M]

        # Creiamo una maschera penalizzante: se i punti appartengono a zone d'asse diverse,
        # abbassiamo drasticamente la loro similarità prima del softmax.
        # Questo impedisce alla matrice P di diventare uniforme!
        similarity_matrix = similarity_matrix - (distanza_asse * 50.0)

        # Ora calcoliamo P: i punti matching saranno vincolati a rimanere nella loro fascia d'asse
        P = F.softmax(similarity_matrix, dim=-1) 

        xyz_b_virtual = torch.bmm(P, xyz_b) 

        # Calcolo dell'importanza di ogni punto basandoci sulla confidenza dei match
        weights = torch.max(P, dim=-1)[0] # [B, N]
        weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-8)
        weights = weights.unsqueeze(-1) # [B, N, 1]

        # Centroide pesato di A e del B virtuale
        centroid_a = torch.sum(xyz_a * weights, dim=1, keepdim=True) # [B, 1, 3]
        centroid_b = torch.sum(xyz_b_virtual * weights, dim=1, keepdim=True) # [B, 1, 3]

        xyz_a_centered = xyz_a - centroid_a
        xyz_b_virtual_centered = xyz_b_virtual - centroid_b

        # Applichiamo i pesi anche alla matrice di covarianza H
        H = torch.bmm((xyz_a_centered * weights).transpose(1, 2), xyz_b_virtual_centered) 

        U, S, V = torch.linalg.svd(H) 
        V_t = V.transpose(1, 2)
        d = torch.det(torch.bmm(V_t, U.transpose(1, 2)))
        
        diag = torch.eye(3, device=frag_a.device).unsqueeze(0).repeat(b, 1, 1)
        diag[:, 2, 2] = d

        R_pred = torch.bmm(torch.bmm(V_t, diag), U.transpose(1, 2)) 
        t_pred = centroid_a.transpose(1, 2) - torch.bmm(R_pred, centroid_b.transpose(1, 2)) 
        t_pred = t_pred.squeeze(-1) # [B, 3]

        # -----------------------------------------------------------------
        # 🆕 ANTIDOTO ALL'EFFETTO PERNO (SHIFT LINEARE DINAMICO)
        # -----------------------------------------------------------------
        # Leggiamo la feature manuale dell'asse Z (canale 6) per capire dove 
        # la rete crede che siano le estremità dell'osso B rispetto ad A.
        z_feat_a = frag_a[..., 6].mean(dim=1) # [B]
        z_feat_b = frag_b[..., 6].mean(dim=1) # [B]

        # Calcoliamo il vettore dell'asse maggiore dell'osso nel sistema ruotato
        # L'asse Z è la terza colonna della matrice di rotazione R_pred
        asse_lunghezza_pred = R_pred[:, :, 2] # [B, 3]

        # Controlliamo l'intensità della traslazione attuale
        norm_t = torch.norm(t_pred, dim=-1, keepdim=True) # [B, 1]
        
        # Se la rete applica una traslazione inferiore a 8mm, significa che è 
        # intrappolata nel perno centrale. Diamo una spinta lungo l'asse!
        mask_bloccato = (norm_t < 8.0).float() # [B, 1]

        # Scegliamo la direzione dello shift basandoci sul segno del disallineamento 
        # delle feature per spingerlo verso la sua metà corretta
        segno_direzione = torch.sign(z_feat_a - z_feat_b).unsqueeze(-1) # [B, 1]
        
        # Generiamo una spinta vigorosa di 25 millimetri lungo l'asse principale
        spinta_lineare = asse_lunghezza_pred * (segno_direzione * 25.0) # [B, 3]

        # Applichiamo lo shift correttivo solo se il modello è bloccato
        t_pred = t_pred + (mask_bloccato * spinta_lineare)
        # -----------------------------------------------------------------

        return R_pred, t_pred

if __name__ == "__main__":
    # Test rapido del nuovo modello SVD (Adattato a 9 canali in ingresso)
    B, N, C = 2, 512, 9
    mock_a = torch.randn(B, N, C)
    mock_b = torch.randn(B, N, C)
    
    model = SVDBoneRegistration(k=16)
    R, t = model(mock_a, mock_b)
    print("🤖 Test dell'Architettura GNN + SVD con Sblocco di Traslazione:")
    print(f" - Matrice di Rotazione R (SVD) predetta: {R.shape}")
    print(f" - Vettore di Traslazione t predetto:     {t.shape}")