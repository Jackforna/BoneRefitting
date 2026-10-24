"""
Griglie di occupazione (dentro/fuori) dei frammenti, per misurare la
compenetrazione REALE tra le ossa.

Il vecchio controllo in env._detect_penetration contava un punto di B
come "dentro" A solo se distava meno di 0.65 mm dal punto piu' vicino
di A: i punti affondati piu' in profondita' nella parete dell'osso
restavano invisibili, e l'agente sfruttava questo buco ottenendo
chamfer e surface migliori della GT facendo compenetrare le ossa.

Qui, per ogni frammento, si calcola UNA VOLTA il winding number
generalizzato della sua mesh originale (data/raw/<nome>.obj) su una
griglia regolare. Il winding number vale ~1 dentro il materiale osseo
e ~0 fuori, e funziona anche su mesh aperte come le scansioni (dove la
frattura lascia bordi non chiusi). Usa SOLO la geometria del singolo
frammento, nessuna informazione sulla posa corretta.

A runtime l'ambiente legge la griglia con interpolazione trilineare:
un lookup per punto, quindi costo trascurabile ad ogni step.

Uso (dalla cartella principale del progetto, dopo preprocess.py):
    python -m srcRL.occupancy
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

# Deve coincidere con il fattore "* 70" applicato ai punti in
# ArchaeologicalIncastroEnv.reset (coordinate normalizzate -> mm).
ENV_SCALE = 70.0

VOXEL_SIZE = 1.0         # mm
GRID_MARGIN = 4.0        # mm attorno al bounding box della mesh
TARGET_FACES = 20000     # semplificazione della mesh prima del calcolo
INSIDE_THRESHOLD = 0.5   # winding number sopra il quale un punto e' "dentro"


def load_mesh_env_frame(obj_path, centroid, radius, target_faces=TARGET_FACES):
    """Carica la mesh, la semplifica e la porta nelle coordinate dell'ambiente (mm)."""
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(obj_path)
    if len(mesh.triangles) > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertices = (vertices - np.asarray(centroid)) / (radius + 1e-8) * ENV_SCALE
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    return vertices, triangles


def winding_numbers(points, vertices, triangles, chunk=128):
    """
    Winding number generalizzato (Jacobson et al. 2013) dei punti
    rispetto alla mesh: somma degli angoli solidi dei triangoli visti
    dal punto, divisa per 4*pi.
    """
    tri = torch.as_tensor(vertices[triangles], dtype=torch.float64)  # (T, 3, 3)
    pts = torch.as_tensor(points, dtype=torch.float64)
    out = []
    for i in range(0, len(pts), chunk):
        p = pts[i:i + chunk, None, None, :]
        rel = tri[None] - p                                  # (C, T, 3, 3)
        a, b, c = rel[:, :, 0], rel[:, :, 1], rel[:, :, 2]
        la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
        det = (a * torch.cross(b, c, dim=-1)).sum(-1)
        den = (la * lb * lc + (a * b).sum(-1) * lc + (b * c).sum(-1) * la + (c * a).sum(-1) * lb)
        out.append(torch.atan2(det, den).sum(-1) / (2 * np.pi))
    # Il segno dipende dall'orientamento delle facce della scansione
    return torch.cat(out).abs().numpy()


def build_occupancy_grid(obj_path, centroid, radius, voxel_size=VOXEL_SIZE, margin=GRID_MARGIN):
    vertices, triangles = load_mesh_env_frame(obj_path, centroid, radius)

    lo = vertices.min(axis=0) - margin
    hi = vertices.max(axis=0) + margin
    shape = np.ceil((hi - lo) / voxel_size).astype(int) + 1
    axes = [lo[d] + np.arange(shape[d]) * voxel_size for d in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    grid_points = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

    print(f"  {os.path.basename(obj_path)}: {len(triangles)} triangoli, griglia {tuple(shape)} "
          f"({len(grid_points)} punti)")
    winding = winding_numbers(grid_points, vertices, triangles).reshape(tuple(shape))

    return {
        "origin": lo.astype(np.float32),
        "voxel_size": np.float32(voxel_size),
        "winding": winding.astype(np.float16),   # indicizzato [x, y, z]
        "env_scale": np.float32(ENV_SCALE),
    }


class OccupancyGrid:
    """Griglia caricata in torch, interrogabile con punti in mm."""

    def __init__(self, data, device="cpu"):
        if float(data["env_scale"]) != ENV_SCALE:
            raise ValueError("Griglia calcolata con una scala diversa da quella dell'ambiente: rigenerala.")
        winding = torch.as_tensor(data["winding"].astype(np.float32), device=device)
        # grid_sample vuole (N, C, D, H, W) con le coordinate in ordine (x, y, z) -> (W, H, D)
        self.volume = winding.permute(2, 1, 0).unsqueeze(0).unsqueeze(0).contiguous()
        self.origin = torch.as_tensor(data["origin"], device=device)
        self.extent = torch.tensor(data["winding"].shape, dtype=torch.float32, device=device) - 1
        self.voxel_size = float(data["voxel_size"])

    @classmethod
    def load(cls, path, device="cpu"):
        return cls(np.load(path), device=device)

    def winding(self, points):
        """Winding number interpolato; 0 (fuori) per i punti esterni alla griglia."""
        idx = (points - self.origin) / self.voxel_size
        norm = idx / self.extent * 2 - 1
        grid = norm.view(1, 1, 1, -1, 3).to(self.volume.dtype)
        values = F.grid_sample(self.volume, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return values.view(-1)

    def inside(self, points, threshold=INSIDE_THRESHOLD):
        return self.winding(points) > threshold


def main(processed_dir="data3/processed", raw_dir="data3/raw", out_dir="data3/processed/occupancy"):
    with open(os.path.join(processed_dir, "normalization.json")) as f:
        norm = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(processed_dir) if f.endswith(".pt"))
    for stem in stems:
        obj_path = os.path.join(raw_dir, f"{stem}.obj")
        if not os.path.exists(obj_path):
            print(f"⚠️ Mesh non trovata per {stem}: {obj_path}")
            continue
        print(f"Calcolo griglia di occupazione per {stem}...")
        data = build_occupancy_grid(obj_path, norm["centroid"], norm["radius"])
        out_path = os.path.join(out_dir, f"{stem}.npz")
        np.savez_compressed(out_path, **data)
        print(f"  salvata in {out_path}")


if __name__ == "__main__":
    main()
