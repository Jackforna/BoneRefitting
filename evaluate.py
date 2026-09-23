import os
import torch
import open3d as o3d
import numpy as np

# Importiamo i nostri moduli dedicati
from src.dataset import BoneFragmentsDataset
from src.models import SVDBoneRegistration

def main():
    # 1. Configurazione hardware e percorsi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "weights/best_model.pth"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Impossibile trovare i pesi del modello in '{model_path}'. Hai già avviato train.py?")

    print(f"🧠 Caricamento del modello GNN + SVD da {model_path}...")
    model = SVDBoneRegistration(k=20)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. Carichiamo un esempio dal Dataset
    print("📦 Estrazione di una coppia di frammenti disallineati per il test...")
    dataset = BoneFragmentsDataset(processed_dir="data/processed/train", num_samples=5)
    sample = dataset[0] # Prendiamo il primo esempio generato casualmente
    
    # Prepariamo i tensori per la rete aggiungendo la dimensione del Batch [B=1, N, 6]
    frag_a, frag_b, T_gt = sample

    # Ora applichi l'unsqueeze (se necessario per simulare il batch size = 1) e sposti sul device:
    frag_a = frag_a.unsqueeze(0).to(device)
    frag_b_disaligned = frag_b.unsqueeze(0).to(device)
    T_gt = T_gt.unsqueeze(0).to(device)

    # 3. Esecuzione dell'IA (Inferenza con SVD differenziabile)
    with torch.no_grad():
        # Il modello restituisce direttamente R (3x3) e t (3)
        R_pred, t_pred = model(frag_a, frag_b_disaligned)
        
        # Portiamo i risultati su CPU e li convertiamo in numpy array
        R_pred = R_pred.squeeze(0).cpu().numpy() # [3, 3]
        t_pred = t_pred.squeeze(0).cpu().numpy() # [3]

    print("\n🔮 Risultati della predizione geometrica analitica (SVD):")
    print(f" - Matrice di Rotazione R stimata:\n{R_pred}")
    print(f" - Vettore di Traslazione t stimato: {t_pred}")

    # 4. Applicazione della trasformazione e visualizzazione 3D
    # Estraiamo i numpy array originari (XYZ) per la visualizzazione
    pts_a = frag_a[0, :, :3].cpu().numpy()
    pts_b_initial = frag_b[:, :3].cpu().numpy() # La posizione iniziale disallineata
    
    # Applichiamo la formula di roto-traslazione classica per allineare i punti
    pts_b_reassembled = (R_pred @ pts_b_initial.T).T + t_pred

    # Creiamo gli oggetti PointCloud di Open3D
    pcd_a = o3d.geometry.PointCloud()
    pcd_a.points = o3d.utility.Vector3dVector(pts_a)
    pcd_a.paint_uniform_color([1, 0.706, 0]) # Arancione per il pezzo fisso A

    pcd_b_final = o3d.geometry.PointCloud()
    pcd_b_final.points = o3d.utility.Vector3dVector(pts_b_reassembled)
    pcd_b_final.paint_uniform_color([0, 0.651, 0.929]) # Azzurro per il pezzo riallineato dall'IA

    print("\n🖥️ Apertura della finestra 3D interattiva finale...")
    print("Esamina l'incastro dei due pezzi. Premi 'q' o chiudi la finestra per terminare.")
    
    # Rendering 3D
    o3d.visualization.draw_geometries(
        [pcd_a, pcd_b_final], 
        window_name="Verifica Incastro Finale - Modello GNN + SVD",
        width=1024, height=768
    )

if __name__ == "__main__":
    main()