import os
import open3d as o3d
import torch
import numpy as np

def preprocess_mesh(obj_path, num_points=2048):
    """
    Carica una mesh .obj, calcola le normali e campiona un numero fisso di punti
    per trasformarla in una Point Cloud.
    """
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"File non trovato: {obj_path}")
        
    print(f"Caricamento in corso: {obj_path}...")
    # Carica la mesh
    mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh.compute_vertex_normals()
    
    # Campionamento uniforme sulla superficie della mesh
    print(f"Campionamento di {num_points} punti dalla superficie...")
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    
    # Estraiamo le coordinate (X, Y, Z) come numpy array
    points = np.asarray(pcd.points, dtype=np.float32)
    normals = np.asarray(pcd.normals, dtype=np.float32)
    
    # Uniamo coordinate e normali in un unico tensore (N, 6)
    # Molte reti neurali 3D (come PointNet++) usano sia la posizione che la normale del punto
    features = np.hstack((points, normals))
    
    return torch.from_numpy(features), pcd

def main():
    # Configurazione percorsi (assumendo di lanciare lo script dalla cartella principale del progetto)
    raw_dir = "data/raw"
    processed_dir = "data/processed/train" # Per ora salviamo tutto in train come test
    os.makedirs(processed_dir, exist_ok=True)
    
    # Lista dei file da processare che hai caricato
    file_names = ["1776a.obj", "2480.obj"]
    
    point_clouds_for_vis = []
    
    for name in file_names:
        obj_path = os.path.join(raw_dir, name)
        
        # Esegui il preprocessing
        tensor_data, pcd_obj = preprocess_mesh(obj_path, num_points=2048)
        
        # Salviamo il file .pt di PyTorch nella cartella processed
        save_name = name.replace(".obj", "_pcd.pt")
        save_path = os.path.join(processed_dir, save_name)
        torch.save(tensor_data, save_path)
        print(f"✅ Salvato tensore PyTorch in: {save_path} (Shape: {tensor_data.shape})\n")
        
        # Teniamo traccia della point cloud per la visualizzazione finale
        point_clouds_for_vis.append(pcd_obj)
        
    # --- VISUALIZZAZIONE INTERATTIVA ---
    print("Apertura finestra 3D. Chiudi la finestra (o premi 'q') per terminare lo script.")
    # Coloriamo i due pezzi campionati per distinguerli nella finestra Open3D
    point_clouds_for_vis[0].paint_uniform_color([1, 0.706, 0])      # Arancione (1776a)
    point_clouds_for_vis[1].paint_uniform_color([0, 0.651, 0.929])  # Azzurro (2480)
    
    o3d.visualization.draw_geometries(point_clouds_for_vis, window_name="Point Clouds Campionate (2048 punti)")

if __name__ == "__main__":
    main()