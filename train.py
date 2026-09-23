import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import open3d as o3d
import numpy as np

# Importiamo i moduli
from src.dataset import BoneFragmentsDataset
from src.models import SVDBoneRegistration
from src.losses import ArchaeologicalIncastroLoss

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, running_chamfer, running_compen = 0.0, 0.0, 0.0
    
    # Variabili di supporto per estrarre l'ultimo allineamento dell'epoca da mostrare
    last_pts_a = None
    last_pts_b_pred = None

    for batch_idx, batch in enumerate(dataloader):
        # Con questa riga singola ed elegante che scompatta la lista in base alla posizione:
        frag_a, frag_b, T_gt = batch

        # Ora sposti i tensori sul device (GPU/CPU) normalmente:
        frag_a = frag_a.to(device)
        frag_b_disaligned = frag_b.to(device)
        T_gt = T_gt.to(device)
                
        optimizer.zero_grad()
        
        # Forward Pass (SVD)
        R_pred, t_pred = model(frag_a, frag_b_disaligned)
        
        t_pred = t_pred.unsqueeze(-1)
        xyz_b = frag_b_disaligned[..., :3]
        nor_b = frag_b_disaligned[..., 3:6]
        
        # Applicazione della roto-traslazione ai punti di B
        xyz_b_transformed = (R_pred @ xyz_b.transpose(1, 2) + t_pred).transpose(1, 2)
        nor_b_transformed = (R_pred @ nor_b.transpose(1, 2)).transpose(1, 2)
        
        frag_b_pred = torch.cat([xyz_b_transformed, nor_b_transformed], dim=-1)
        
        # Calcolo Loss
        loss, chamfer, compenetration = criterion(frag_a, frag_b_pred)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_chamfer += chamfer.item()
        running_compen += compenetration.item()
        
        # Salviamo l'ultimissimo sample del batch per passarlo al visualizzatore 3D
        if batch_idx == len(dataloader) - 1:
            last_pts_a = frag_a[0, :, :3].detach().cpu().numpy()
            last_pts_b_pred = frag_b_pred[0, :, :3].detach().cpu().numpy()
            
    return (running_loss / len(dataloader), 
            running_chamfer / len(dataloader), 
            running_compen / len(dataloader), 
            last_pts_a, last_pts_b_pred)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0001)
    # Aggiungiamo un flag per decidere se resettare o riprendere
    parser.add_argument("--resume", action="store_true", help="Riprendi dall'ultimo checkpoint salvato")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = BoneFragmentsDataset(processed_dir="data/processed/train", num_samples=100)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    model = SVDBoneRegistration(k=20).to(device)
    
    # --- 🔄 BLOCCO PER RIPRENDERE L'ADDESTRAMENTO ---
    model_path = "weights/best_model.pth"
    if args.resume and os.path.exists(model_path):
        print(f"📥 Trovato modello precedente in '{model_path}'. Caricamento dei pesi in corso...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Pesi caricati con successo! L'addestramento riprenderà da questa base.")
    elif args.resume:
        print(f"⚠️ Flag --resume attivo, ma non ho trovato nessun file in '{model_path}'. Parto da zero.")
    # ------------------------------------------------
    
    criterion = ArchaeologicalIncastroLoss(tolleranza_contatto=10, tolleranza_compenetrazione=0.6)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # ... (il resto del codice dell'inizializzazione di Open3D rimane identico)
    
    # --- 🖥️ INIZIALIZZAZIONE DELLA FINESTRA 3D DI OPEN3D (NON-BLOCCANTE) ---
    print("\n🖥️ Inizializzazione della finestra di tracciamento 3D...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Tracciamento Incastro Real-time", width=800, height=600)
    
    # Creiamo gli oggetti PointCloud vuoti
    pcd_a = o3d.geometry.PointCloud()
    pcd_b = o3d.geometry.PointCloud()
    
    # Assegniamo colori fissi (Arancione per A, Azzurro per B)
    pcd_a.paint_uniform_color([1, 0.706, 0])
    pcd_b.paint_uniform_color([0, 0.651, 0.929])
    
    # Aggiungiamo le geometrie al visualizzatore (per ora sono vuote)
    vis.add_geometry(pcd_a)
    vis.add_geometry(pcd_b)
    
    first_epoch = True
    best_loss = float('inf')
    # ------------------------------------------------------------------------

    for epoch in range(1, args.epochs + 1):
        loss, chamfer, compen, pts_a, pts_b_pred = train_epoch(model, train_loader, criterion, optimizer, device)
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] -> Loss: {loss:.4f} | Chamfer: {chamfer:.4f} | Compen: {compen:.4f}")
        
        # --- 🔄 AGGIORNAMENTO DELLA VISUALIZZAZIONE 3D ---
        if pts_a is not None and pts_b_pred is not None:
            # Aggiorniamo i vettori dei punti con i nuovi dati calcolati in questa epoca
            pcd_a.points = o3d.utility.Vector3dVector(pts_a)
            pcd_b.points = o3d.utility.Vector3dVector(pts_b_pred)
            
            # Coloriamo i punti stabili ad ogni epoca (Giallo e Azzurro)
            pcd_a.paint_uniform_color([1, 0.706, 0])
            pcd_b.paint_uniform_color([0, 0.651, 0.929])
            
            # Comunichiamo a Open3D che i dati geometrici sono cambiati
            vis.update_geometry(pcd_a)
            vis.update_geometry(pcd_b)
            
            # Se è la prima epoca, resettiamo la telecamera per centrare l'osso
            if first_epoch:
                vis.reset_view_point(True)
                first_epoch = False
                
            # Eseguiamo il rendering del frame
            vis.poll_events()
            vis.update_renderer()
        # --------------------------------------------------

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), "weights/best_model.pth")

    # Alla fine dell'addestramento, distruggiamo la finestra in modo pulito
    print("🏁 Addestramento completato. Chiusura del visualizzatore.")
    vis.destroy_window()

if __name__ == "__main__":
    main()