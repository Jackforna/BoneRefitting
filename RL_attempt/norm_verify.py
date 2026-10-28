import torch
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import BoneFragmentsDataset

# =====================================================================
# 🧠 FUNZIONE DI STIMA GEOMETRICA DELLE NORMALI (Pure PyTorch)
# =====================================================================
def estimate_normals(points, k=15):
    """
    Calcola le normali geometriche reali per ogni punto usando la PCA locale.
    Punta i vettori verso l'esterno dell'osso (a raggiera).
    """
    with torch.no_grad():
        device = points.device
        dtype = points.dtype
        
        # 1. Matrice delle distanze tra tutti i punti
        dists = torch.cdist(points, points)
        
        # 2. Trova i k punti più vicini per ogni punto
        _, knn_idx = torch.topk(dists, k, dim=1, largest=False)
        
        # 3. Raggruppa i vicini geometrici
        knn_points = points[knn_idx] 
        
        # 4. Centra i vicini attorno alla loro media locale
        means = torch.mean(knn_points, dim=1, keepdim=True)
        centered = knn_points - means
        
        # 5. Matrice di covarianza locale per ogni punto
        cov = torch.bmm(centered.transpose(1, 2), centered) / (k - 1)
        
        # 6. Decomposizione in autovalori
        _, eigenvectors = torch.linalg.eigh(cov)
        normals = eigenvectors[:, :, 0] # Il primo autovettore è la normale
        
        # 7. ORIENTAMENTO ESTERNO (A RAGGIERA)
        centro_globale = torch.mean(points, dim=0, keepdim=True)
        vettori_dal_centro = points - centro_globale
        
        # Se punta in dentro (prodotto scalare < 0), invertiamo il verso
        dot_products = torch.sum(normals * vettori_dal_centro, dim=1, keepdim=True)
        flips = torch.where(dot_products < 0, -1.0, 1.0)
        normals = normals * flips
        
        # Normalizzazione finale
        normals = normals / torch.norm(normals, dim=1, keepdim=True).clamp(min=1e-8)
        
        return normals

# =====================================================================
# 📦 CARICAMENTO DEL DATASET (Adatta le righe qui sotto!)
# =====================================================================
# 👇 CONTROLLA QUESTE RIGHE: devono essere identiche a come carichi il dataset nel train
# from dataset_file import BoneFragmentsDataset
# dataset = BoneFragmentsDataset(root_dir="path/ai/tuoi/dati")

cartella_elaborati = "data/processed/train"
dataset = BoneFragmentsDataset(
        processed_dir=cartella_elaborati, 
        num_samples=2000, 
        max_translation=3.0, 
        max_rotation=10.0 
    )

print("Caricamento del frammento dal dataset...")
frag_a_full, frag_b_full, _ = dataset[0]

# Prendiamo solo i punti XYZ originari (canali 0, 1, 2) che sono corretti al 100%
frag_a = frag_a_full[:, :3].clone().detach()
frag_b_perfect = frag_b_full[:, :3].clone().detach()

# =====================================================================
# ✨ CALCOLO DELLE NUOVE NORMALI GEOMETRICHE (Per entrambi i pezzi)
# =====================================================================
print("Ricalcolo geometrico delle normali per l'osso A e l'osso B...")
nuove_normali_a = estimate_normals(frag_a, k=15)
nuove_normali_b = estimate_normals(frag_b_perfect, k=15)

with torch.no_grad():
    # Ricalcoliamo le distanze e la maschera di prossimità (zona della frattura)
    dists = torch.cdist(frag_b_perfect, frag_a)
    min_dist_b_to_a, static_indices = torch.min(dists, dim=1)
    
    # Selezioniamo il 15% dei punti più vicini (l'area di contatto)
    num_punti_b = frag_b_perfect.shape[0]
    k_vicini = int(num_punti_b * 0.15)
    valori_ordinati, indici_ordinati = torch.sort(min_dist_b_to_a)
    
    maschera_frattura = torch.zeros_like(min_dist_b_to_a, dtype=torch.bool)
    maschera_frattura[indici_ordinati[:k_vicini]] = True
    
    # Associazione delle normali di A ai punti corrispondenti di B
    normals_a_matched = nuove_normali_a[static_indices]
    
    # 1. Calcolo Allineamento Originale (Incastro di riferimento)
    dot_perfect = torch.sum(nuove_normali_b[maschera_frattura] * normals_a_matched[maschera_frattura], dim=1)
    align_perfect = -torch.mean(dot_perfect).item()
    
    # =====================================================================
    # 🔄 TEST DEI TRE ASSI DI ROTAZIONE (X, Y, Z) a 180°
    # =====================================================================
    theta = np.pi # 180 gradi
    
    # Matrice di Rotazione attorno a X
    R_X = torch.tensor([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ], dtype=frag_b_perfect.dtype)
    
    # Matrice di Rotazione attorno a Y
    R_Y = torch.tensor([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ], dtype=frag_b_perfect.dtype)
    
    # Matrice di Rotazione attorno a Z
    R_Z = torch.tensor([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ], dtype=frag_b_perfect.dtype)
    
    print("\n" + "="*50)
    print("🔄 VERIFICA ANALITICA DELLE ROTAZIONI SUI TRE ASSI")
    print("="*50)
    print(f"Posizione originale (Incastro GT):       {align_perfect:.4f}")
    print("-"*50)
    
    # Test asse X
    norm_X = torch.mm(nuove_normali_b, R_X.t())
    align_X = -torch.mean(torch.sum(norm_X[maschera_frattura] * normals_a_matched[maschera_frattura], dim=1)).item()
    print(f"Dopo rotazione 180° su X:                {align_X:.4f}  (Delta: {abs(align_perfect - align_X):.4f})")
    
    # Test asse Y
    norm_Y = torch.mm(nuove_normali_b, R_Y.t())
    align_Y = -torch.mean(torch.sum(norm_Y[maschera_frattura] * normals_a_matched[maschera_frattura], dim=1)).item()
    print(f"Dopo rotazione 180° su Y:                {align_Y:.4f}  (Delta: {abs(align_perfect - align_Y):.4f})")
    
    # Test asse Z
    norm_Z = torch.mm(nuove_normali_b, R_Z.t())
    align_Z = -torch.mean(torch.sum(norm_Z[maschera_frattura] * normals_a_matched[maschera_frattura], dim=1)).item()
    print(f"Dopo rotazione 180° su Z:                {align_Z:.4f}  (Delta: {abs(align_perfect - align_Z):.4f})")
    print("="*50 + "\n")

# =====================================================================
# 🎨 ISPEZIONE VISIVA 3D (MATPLOTLIB)
# =====================================================================
print("Generazione del grafico 3D... Chiudi la finestra per terminare.")

step_campionamento = 50
punti_np = frag_b_perfect.numpy()[::step_campionamento]
normali_np = nuove_normali_b.numpy()[::step_campionamento]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(punti_np[:, 0], punti_np[:, 1], punti_np[:, 2], 
           c='blue', s=20, label='Punti Osso B')

ax.quiver(punti_np[:, 0], punti_np[:, 1], punti_np[:, 2],
          normali_np[:, 0], normali_np[:, 1], normali_np[:, 2],
          length=2.5, color='red', normalize=True, label='Nuove Normali Geometriche')

ax.set_title("Verifica delle Nuove Normali PCA Stimate")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.legend()
plt.show()