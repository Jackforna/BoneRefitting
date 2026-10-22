"""
Fracture mask per mesh CHIUSE (senza bordi aperti), dove preprocess.py non
puo' usare il bordo della scansione.

Il vecchio metodo di riserva sceglieva i punti con rugosita' + curvatura
alte: sulle mesh chiuse prende soprattutto gli spigoli laterali dell'osso,
curvi ma lisci. Qui conta solo la rugosita':
  1. per ogni punto campionato, varianza delle normali della MESH DENSA in
     un raggio di ROUGHNESS_RADIUS (circa 4 mm): alta sulla superficie di
     rottura, frastagliata; bassa sulla corticale e sugli spigoli lisci;
  2. il valore viene mediato sui SMOOTH_NEIGHBORS punti vicini, per togliere
     i picchi isolati (rumore di scansione);
  3. la maschera e' il MASK_FRACTION dei punti piu' rugosi.
Usa solo la geometria del singolo frammento.
"""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

ROUGHNESS_RADIUS = 0.055   # coordinate normalizzate (~4 mm per ossa di ~70 mm di raggio comune)
VOXEL = 0.004              # sottocampionamento della mesh densa (~0.3 mm)
MIN_NEIGHBORS = 12
SMOOTH_NEIGHBORS = 20
MASK_FRACTION = 0.15


def roughness_score(mesh, points_norm, centroid, radius):
    """Varianza delle normali della mesh densa attorno a ogni punto (coordinate normalizzate)."""
    mesh = o3d.geometry.TriangleMesh(mesh)
    mesh.compute_vertex_normals()
    pc = o3d.geometry.PointCloud(mesh.vertices)
    pc.normals = mesh.vertex_normals
    verts = (np.asarray(pc.points) - centroid) / radius
    pc.points = o3d.utility.Vector3dVector(verts)
    pc = pc.voxel_down_sample(VOXEL)
    V, N = np.asarray(pc.points), np.asarray(pc.normals)
    tree = cKDTree(V)
    score = np.empty(len(points_norm), dtype=np.float32)
    for i, (p, ids) in enumerate(zip(points_norm, tree.query_ball_point(points_norm, ROUGHNESS_RADIUS))):
        if len(ids) < MIN_NEIGHBORS:
            ids = tree.query(p, k=MIN_NEIGHBORS)[1]
        score[i] = np.var(N[ids], axis=0).sum()
    return score


def extract_fracture_mask_closed(mesh, points_norm, centroid, radius):
    score = roughness_score(mesh, points_norm, centroid, radius)
    nbrs = cKDTree(points_norm).query(points_norm, k=SMOOTH_NEIGHBORS)[1]
    smooth = score[nbrs].mean(axis=1)
    mask = smooth >= np.percentile(smooth, 100 * (1 - MASK_FRACTION))
    return mask.astype(np.float32), smooth
