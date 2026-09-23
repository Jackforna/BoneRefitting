import os
import torch
from torch.utils.data import Dataset
import numpy as np

class BoneFragmentsDataset(Dataset):
    def __init__(self, processed_dir, num_samples=1000, max_translation=50.0, max_rotation=180.0):
        """
        processed_dir: Cartella dove abbiamo salvato i file .pt (es. data/processed/train)
        num_samples: Quanti "esempi" virtuali di disallineamento vogliamo generare per epoca
        max_translation: Massima traslazione casuale in millimetri
        max_rotation: Massima rotazione casuale in gradi
        """
        self.processed_dir = processed_dir
        self.num_samples = num_samples
        self.max_translation = max_translation
        self.max_rotation = np.radians(max_rotation) # Convertiamo in radianti
        
        # Carichiamo i tensori geometrici salvati dal preprocessore
        self.path_a = os.path.join(processed_dir, "1776a_pcd.pt")
        self.path_b = os.path.join(processed_dir, "2480_pcd.pt")
        
        if not os.path.exists(self.path_a) or not os.path.exists(self.path_b):
            raise FileNotFoundError("Assicurati di aver eseguito preprocess.py prima di usare il dataset!")
            
        # Carichiamo i dati (Shape: [8192, 6] -> XYZ + Normali)
        raw_data_a = torch.load(self.path_a)
        raw_data_b = torch.load(self.path_b)

        # 👑 APPLICHIAMO LA PCA GLOBALE DI COPPIA PER PRESERVARE LA DISTANZA DI INCASTRO
        self.data_a, self.data_b = self._apply_global_pca_alignment(raw_data_a, raw_data_b)

    def __len__(self):
        return self.num_samples
    
    def _apply_global_pca_alignment(self, tensor_a, tensor_b):
        """Centra e allinea i frammenti usando un sistema di riferimento globale comune"""
        xyz_a = tensor_a[:, :3].numpy()
        nor_a = tensor_a[:, 3:].numpy()
        xyz_b = tensor_b[:, :3].numpy()
        nor_b = tensor_b[:, 3:].numpy()

        # 1. Troviamo il baricentro dell'INTERO sistema ricomposto (Unione di A e B)
        combined_xyz = np.vstack([xyz_a, xyz_b])
        global_centroid = np.mean(combined_xyz, axis=0)

        # Centriamo entrambi i frammenti rispetto allo STESSO punto globale
        xyz_a_centered = xyz_a - global_centroid
        xyz_b_centered = xyz_b - global_centroid
        combined_centered = combined_xyz - global_centroid

        # 2. Calcoliamo la PCA sull'intero osso unito per trovare l'asse maggiore globale
        cov = np.cov(combined_centered.T)
        autovalori, autovettori = np.linalg.eigh(cov)
        idx = np.argsort(autovalori)[::-1]
        R_pca = autovettori[:, idx]

        if np.linalg.det(R_pca) < 0:
            R_pca[:, -1] *= -1

        # Allineiamo punti e normali usando la stessa trasformazione rigida globale
        xyz_a_aligned = xyz_a_centered @ R_pca
        nor_a_aligned = nor_a @ R_pca
        xyz_b_aligned = xyz_b_centered @ R_pca
        nor_b_aligned = nor_b @ R_pca

        # 3. CALCOLO DELLE FEATURE CON LIMITI GLOBALI
        # Troviamo il min e max assoluto dell'asse Z dell'intero osso lungo il fusto
        combined_aligned_z = np.concatenate([xyz_a_aligned[:, 2], xyz_b_aligned[:, 2]])
        z_min, z_max = np.min(combined_aligned_z), np.max(combined_aligned_z)
        z_range = z_max - z_min + 1e-8

        # Generazione feature per il Frammento A
        z_norm_a = (xyz_a_aligned[:, 2] - z_min) / z_range
        raggio_a = np.sqrt(xyz_a_aligned[:, 0]**2 + xyz_a_aligned[:, 1]**2)
        raggio_norm_a = (raggio_a - np.min(raggio_a)) / (np.max(raggio_a) - np.min(raggio_a) + 1e-8)
        estremita_a = np.abs(z_norm_a - 0.5) * 2.0

        # Generazione feature per il Frammento B
        z_norm_b = (xyz_b_aligned[:, 2] - z_min) / z_range
        raggio_b = np.sqrt(xyz_b_aligned[:, 0]**2 + xyz_b_aligned[:, 1]**2)
        raggio_norm_b = (raggio_b - np.min(raggio_b)) / (np.max(raggio_b) - np.min(raggio_b) + 1e-8)
        estremita_b = np.abs(z_norm_b - 0.5) * 2.0

        # Composizione dei 9 canali finali
        features_a = np.hstack([xyz_a_aligned, nor_a_aligned, z_norm_a[:, None], raggio_norm_a[:, None], estremita_a[:, None]])
        features_b = np.hstack([xyz_b_aligned, nor_b_aligned, z_norm_b[:, None], raggio_norm_b[:, None], estremita_b[:, None]])

        return torch.from_numpy(features_a).float(), torch.from_numpy(features_b).float()

    def __getitem__(self, idx):
        # 1. Separazione dei 9 canali di B (Ground Truth preservato nella sua vera coordinata staccata)
        xyz_b_gt = self.data_b[:, :3]        
        nor_b_gt = self.data_b[:, 3:6]       
        extra_feats_b = self.data_b[:, 6:]   

        # 2. Generazione del disallineamento casuale di addestramento
        angles = np.random.uniform(-self.max_rotation, self.max_rotation, size=3)
        cx, cy, cz = np.cos(angles)
        sx, sy, sz = np.sin(angles)

        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

        R = torch.from_numpy(Rz @ Ry @ Rx).float()
        
        # Generiamo un disturbo spaziale puro tridimensionale
        t = torch.from_numpy(np.random.uniform(-self.max_translation, self.max_translation, size=(3,))).float()

        # 3. Applicazione del disallineamento
        xyz_b_disaligned = (R @ xyz_b_gt.t()).t() + t
        nor_b_disaligned = (R @ nor_b_gt.t()).t()

        # 4. Ricomposizione
        frag_b_disaligned = torch.cat([xyz_b_disaligned, nor_b_disaligned, extra_feats_b], dim=-1)
        frag_a = self.data_a

        T_gt = torch.eye(4)
        T_gt[:3, :3] = R
        T_gt[:3, 3] = t

        return frag_a, frag_b_disaligned, T_gt
    
if __name__ == "__main__":
    # Test rapido del dataset (corretto per evitare crash da tuple)
    try:
        dataset = BoneFragmentsDataset(processed_dir="data/processed/train", num_samples=5)
        print(f"📊 Dataset inizializzato con successo. Numero di campioni virtuali: {len(dataset)}")
        
        frag_a, frag_b, T_gt = dataset[0]
        print("\n🔍 Analisi delle dimensioni dell'output:")
        print(f" - Frammento A (Fisso):        {frag_a.shape}")
        print(f" - Frammento B (Disallineato): {frag_b.shape}")
        print(f" - Matrice di Trasformazione:  {T_gt.shape}")
        print("\n✅ Il dataset funziona correttamente ed è pronto per PyTorch!")
    except Exception as e:
        print(f"❌ Errore durante il test del dataset: {e}")