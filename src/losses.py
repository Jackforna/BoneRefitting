import torch
import torch.nn as nn

class ArchaeologicalIncastroLoss(nn.Module):
    def __init__(self, tolleranza_contatto=3.0, tolleranza_compenetrazione=0.6):
        """
        tolleranza_contatto: Ridotta a 3.0mm (10mm creava l'effetto buco nero).
        tolleranza_compenetrazione: Sotto i 0.6mm scatta il muro invisibile.
        """
        super(ArchaeologicalIncastroLoss, self).__init__()
        self.tol_contatto = tolleranza_contatto
        self.barriera = tolleranza_compenetrazione

    def forward(self, frag_a, frag_b_pred):
        xyz_a = frag_a[..., :3]      # [B, N, 3]
        xyz_b = frag_b_pred[..., :3] # [B, M, 3]

        batch_size = xyz_a.shape[0]

        # 1. REWARD ASSIALE (Ottima per l'orientamento degli assi principali)
        loss_asse = 0.0
        for b in range(batch_size):
            pts_a_cent = xyz_a[b] - torch.mean(xyz_a[b], dim=0)
            pts_b_cent = xyz_b[b] - torch.mean(xyz_b[b], dim=0)
            
            cov_a = torch.mm(pts_a_cent.t(), pts_a_cent)
            cov_b = torch.mm(pts_b_cent.t(), pts_b_cent)
            
            _, _, V_a = torch.linalg.svd(cov_a)
            _, _, V_b = torch.linalg.svd(cov_b)
            
            asse_a = V_a[:, 0] 
            asse_b = V_b[:, 0] 
            
            parallelismo = torch.abs(torch.dot(asse_a, asse_b))
            loss_asse += (1.0 - parallelismo)
        
        loss_asse = loss_asse / batch_size

        # Calcolo della matrice delle distanze Euclidee
        dist_matrix = torch.cdist(xyz_a, xyz_b, p=2) # [B, N, M]

        # Troviamo le distanze minime nei due sensi
        min_dist_b_to_a, _ = torch.min(dist_matrix, dim=1) # [B, M]
        #min_dist_a_to_b, _ = torch.min(dist_matrix, dim=2) # [B, N]

        # Se un punto è più lontano di 8mm, non trascinare la rete con gradienti enormi
        dist_b_to_a_tagliata = torch.clamp(min_dist_b_to_a, max=8.0)
        loss_chamfer_globale = torch.mean(dist_b_to_a_tagliata)

        # 👑 STRATEGICO: VERA CHAMFER LOSS GLOBALE
        # Fornisce un gradiente liscio e continuo in tutto lo spazio 3D. 
        # Dice alla rete come traslare i pezzi anche quando sono lontanissimi.
        #loss_chamfer_globale = torch.mean(min_dist_b_to_a) + torch.mean(min_dist_a_to_b)

        # 2. REWARD PUNTI A CONTATTO (Ora molto più selettiva grazie ai 3mm)
        punti_a_contatto = (min_dist_b_to_a > self.barriera) & (min_dist_b_to_a <= self.tol_contatto)
        ratio_contatto = torch.sum(punti_a_contatto.float(), dim=-1) / xyz_b.shape[1]
        loss_contatto = torch.mean(1.0 - ratio_contatto)

        # 3. MURO DI COMPENETRAZIONE STABILE (Normalizzato sulla media)
        violazione = self.barriera - min_dist_b_to_a
        punti_violati_mask = violazione > 0.0
        
        if punti_violati_mask.any():
            # Usiamo la somma delle violazioni
            penalita_compenetrazione = torch.sum(violazione[punti_violati_mask] ** 2) /batch_size
        else:
            penalita_compenetrazione = torch.tensor(0.0, device=frag_a.device, requires_grad=True)

        # NUOVO BILANCIAMENTO DEI PESI
        # La chamfer globale guida il macro-movimento, l'asse ruota, la compenetrazione fa da scudo duro
        total_loss = (10.0 * loss_asse) + (2.0 * loss_chamfer_globale) + (500.0 * penalita_compenetrazione)

        # Nota: restituiamo comunque loss_contatto come secondo parametro per non rompere il tuo train.py
        # che lo stampa a schermo chiamandolo (erroneamente) "Chamfer"
        return total_loss, loss_chamfer_globale, penalita_compenetrazione