import os
import json

import open3d as o3d
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.cluster import DBSCAN
from torch_geometric.data import Data

NUM_POINTS = 2048
KNN = 16
K_NEIGHBORS_SCORE = 20

# --- Parametri estrazione fracture mask -----------------------------------

# Distanza massima (coordinate normalizzate) da un vertice di bordo
# aperto perche' un punto sia CANDIDATO (necessario ma non piu'
# sufficiente da solo).
FRACTURE_BOUNDARY_RADIUS = 0.05

# Percentile dello score (rugosita' + curvatura), calcolato SOLO tra
# i candidati di bordo, sopra il quale un candidato diventa seme.
# Isola la porzione irregolare/frastagliata dell'anello di bordo
# (vera frattura) dalla porzione piu' regolare (es. apertura naturale
# della cavita' interna, su frammenti tubolari).
BOUNDARY_SEED_PERCENTILE = 55

# Percentile globale dello score, usato SOLO come fallback se la mesh
# e' watertight (nessun bordo aperto rilevabile).
FALLBACK_SEED_PERCENTILE = 70

DBSCAN_EPS = 0.05
DBSCAN_MIN_SAMPLES = 4
MIN_CLUSTER_SIZE = 8

# Percentile globale dello score sopra il quale un punto puo' essere
# aggiunto durante il region growing (soglia piu' permissiva del seed,
# per riempire i buchi tra i semi).
GROWTH_PERCENTILE = 50

# Frazione massima della mesh che la maschera puo' arrivare a coprire.
MAX_FRACTURE_RATIO = 0.30

# Durante il region growing, se e' disponibile il bordo aperto, un
# punto puo' essere aggiunto solo se resta entro questa distanza dal
# bordo (multiplo di FRACTURE_BOUNDARY_RADIUS). Impedisce alla crescita
# di "sconfinare" di nuovo su tutta la superficie corticale, che era
# il problema della versione senza vincolo di bordo.
GROWTH_MAX_BOUNDARY_DIST = FRACTURE_BOUNDARY_RADIUS * 2.5


def build_knn_graph(points, k=16):
    nbrs = NearestNeighbors(n_neighbors=k + 1)
    nbrs.fit(points)
    _, indices = nbrs.kneighbors(points)

    rows, cols = [], []
    for i in range(len(points)):
        for j in indices[i][1:]:
            rows.append(i)
            cols.append(j)
            rows.append(j)
            cols.append(i)

    return torch.tensor([rows, cols], dtype=torch.long)


def extract_boundary_vertex_positions(mesh):
    """
    Ritorna le posizioni dei vertici sui bordi aperti della mesh
    (edge condivisi da un solo triangolo), oppure None se la mesh e'
    watertight.

    Segnale topologico: se il frammento e' stato scansionato senza
    includere la superficie di frattura come faccia chiusa, il bordo
    aperto coincide col perimetro della frattura -- molto piu'
    affidabile della sola curvatura, che confonde rumore di scansione
    con rottura reale.
    """

    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    edge_count = {}
    for tri in triangles:
        edges = (
            tuple(sorted((int(tri[0]), int(tri[1])))),
            tuple(sorted((int(tri[1]), int(tri[2])))),
            tuple(sorted((int(tri[2]), int(tri[0])))),
        )
        for e in edges:
            edge_count[e] = edge_count.get(e, 0) + 1

    boundary_vertex_idx = set()
    for edge, count in edge_count.items():
        if count == 1:
            boundary_vertex_idx.add(edge[0])
            boundary_vertex_idx.add(edge[1])

    if len(boundary_vertex_idx) == 0:
        return None

    return vertices[sorted(boundary_vertex_idx)].astype(np.float32)


def compute_roughness_and_curvature(points, normals, k=K_NEIGHBORS_SCORE):
    """
    Per ogni punto: rugosita' (varianza delle normali nell'intorno,
    alta su superfici frastagliate) e curvatura (rapporto autovalori
    della covarianza locale). Un'unica query KNN vettorizzata invece
    di una query per punto.
    """

    N = len(points)
    tree = KDTree(points)
    _, idx = tree.query(points, k=k)

    roughness = np.zeros(N, dtype=np.float32)
    curvature = np.zeros(N, dtype=np.float32)

    for i in range(N):
        nbr_pts = points[idx[i]]
        nbr_norms = normals[idx[i]]

        roughness[i] = np.var(nbr_norms, axis=0).sum()

        centered = nbr_pts - nbr_pts.mean(axis=0)
        cov = centered.T @ centered / (k - 1)
        eigvals = np.linalg.eigvalsh(cov)
        curvature[i] = eigvals[0] / (eigvals.sum() + 1e-8)

    return roughness, curvature


def extract_fracture_mask(points, normals, boundary_vertices=None):
    """
    Stima non supervisionata della superficie di frattura, in tre fasi:

    1) CANDIDATI: se disponibile il bordo aperto (mesh non watertight),
       i candidati sono i punti vicini al bordo -- segnale topologico
       forte, ma su frammenti tubolari puo' includere un intero anello
       (vera frattura + apertura naturale della cavita' interna).
       Fallback (mesh watertight): candidati ad alto score globale.

    2) SEMI: tra i candidati, teniamo solo quelli con score
       (rugosita' + curvatura) elevato RELATIVAMENTE AGLI ALTRI
       CANDIDATI (non a tutta la mesh) -- isola la porzione
       irregolare dell'anello, quella vera. Poi DBSCAN, tenendo
       TUTTI i cluster validi (non solo il piu' grande): la frattura
       reale puo' essere composta da piu' patch separati.

    3) REGION GROWING vincolato: espande i semi ai vicini con score
       sopra una soglia piu' permissiva, MA solo se restano vicini al
       bordo aperto (quando disponibile) -- impedisce alla crescita
       di dilagare di nuovo su tutta la superficie corticale.
    """

    N = len(points)
    roughness, curvature = compute_roughness_and_curvature(points, normals)
    score = roughness + 2.5 * curvature

    boundary_dist = None

    if boundary_vertices is not None and len(boundary_vertices) > 0:
        tree_b = KDTree(boundary_vertices)
        boundary_dist, _ = tree_b.query(points, k=1)
        boundary_dist = boundary_dist.reshape(-1)

        candidate_idx = np.where(boundary_dist < FRACTURE_BOUNDARY_RADIUS)[0]

        if len(candidate_idx) == 0:
            seed_threshold = np.percentile(score, FALLBACK_SEED_PERCENTILE)
            candidate_idx = np.where(score >= seed_threshold)[0]
        else:
            cand_score = score[candidate_idx]
            seed_threshold = np.percentile(cand_score, BOUNDARY_SEED_PERCENTILE)
            candidate_idx = candidate_idx[cand_score >= seed_threshold]

    else:
        seed_threshold = np.percentile(score, FALLBACK_SEED_PERCENTILE)
        candidate_idx = np.where(score >= seed_threshold)[0]

    mask = np.zeros(N, dtype=bool)

    if len(candidate_idx) == 0:
        return mask.astype(np.float32), curvature

    candidate_pts = points[candidate_idx]
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(candidate_pts)
    labels = db.labels_

    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    valid_clusters = unique_labels[counts >= MIN_CLUSTER_SIZE]

    if len(valid_clusters) == 0:
        mask[candidate_idx] = True
        return mask.astype(np.float32), curvature

    seed_idx = candidate_idx[np.isin(labels, valid_clusters)]
    mask[seed_idx] = True

    tree_pts = KDTree(points)
    queue = list(seed_idx)
    growth_threshold = np.percentile(score, GROWTH_PERCENTILE)
    max_allowed = int(N * MAX_FRACTURE_RATIO)

    while queue and mask.sum() < max_allowed:
        curr = queue.pop(0)
        nbrs = tree_pts.query([points[curr]], k=8, return_distance=False)[0]

        for nb in nbrs:
            if mask[nb]:
                continue
            if score[nb] < growth_threshold:
                continue
            if boundary_dist is not None and boundary_dist[nb] > GROWTH_MAX_BOUNDARY_DIST:
                continue

            mask[nb] = True
            queue.append(nb)

    return mask.astype(np.float32), curvature


def load_fragment(obj_path, num_points=NUM_POINTS):
    if not os.path.exists(obj_path):
        raise FileNotFoundError(obj_path)

    print(f"Loading {obj_path}")
    mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh.compute_vertex_normals()

    boundary_vertices = extract_boundary_vertex_positions(mesh)

    pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.orient_normals_consistent_tangent_plane(50)

    points = np.asarray(pcd.points, dtype=np.float32)
    normals = np.asarray(pcd.normals, dtype=np.float32)

    return points, normals, pcd, boundary_vertices


def build_graph(points, normals, curvature, fracture_mask, centroid_distance):
    """
    Layout a 9 colonne, IDENTICO a quello atteso da model.py
    (SAGEConv(9, 64), FRACTURE_MASK_COL = 7) e da environment.py
    (frag_full[:, 7] come maschera): points(3) + normals(3) +
    curvature(1) + fracture_mask(1) + centroid_distance(1).

    Se in futuro cambi questo layout, aggiorna ANCHE model.py e
    environment.py: usano indici di colonna hard-coded, quindi un
    disallineamento e' silenzioso (nessun errore, solo maschere
    lette a caso).
    """

    features = np.column_stack([
        points,              # 0,1,2
        normals,             # 3,4,5
        curvature,            # 6
        fracture_mask,        # 7
        centroid_distance,    # 8
    ])

    x = torch.tensor(features, dtype=torch.float32)
    edge_index = build_knn_graph(points, k=KNN)

    return Data(x=x, edge_index=edge_index)


def preprocess_pair(path_a, path_b, num_points=NUM_POINTS):
    points_a, normals_a, pcd_a, boundary_a = load_fragment(path_a, num_points)
    points_b, normals_b, pcd_b, boundary_b = load_fragment(path_b, num_points)

    joint_points = np.concatenate([points_a, points_b], axis=0)
    centroid = joint_points.mean(axis=0)
    radius = np.linalg.norm(joint_points - centroid, axis=1).max()

    points_a_norm = (points_a - centroid) / (radius + 1e-8)
    points_b_norm = (points_b - centroid) / (radius + 1e-8)

    # I vertici di bordo vanno normalizzati con LO STESSO centroide e
    # raggio dei punti campionati, altrimenti FRACTURE_BOUNDARY_RADIUS
    # non e' nella stessa scala.
    boundary_a_norm = (boundary_a - centroid) / (radius + 1e-8) if boundary_a is not None else None
    boundary_b_norm = (boundary_b - centroid) / (radius + 1e-8) if boundary_b is not None else None

    fracture_mask_a, curvature_a = extract_fracture_mask(points_a_norm, normals_a, boundary_a_norm)
    fracture_mask_b, curvature_b = extract_fracture_mask(points_b_norm, normals_b, boundary_b_norm)

    centroid_dist_a = np.linalg.norm(points_a_norm, axis=1).astype(np.float32)
    centroid_dist_b = np.linalg.norm(points_b_norm, axis=1).astype(np.float32)

    graph_a = build_graph(points_a_norm, normals_a, curvature_a, fracture_mask_a, centroid_dist_a)
    graph_b = build_graph(points_b_norm, normals_b, curvature_b, fracture_mask_b, centroid_dist_b)

    normalization = {"centroid": centroid.tolist(), "radius": float(radius)}

    return graph_a, graph_b, pcd_a, pcd_b, normalization


def colorize_by_fracture_mask(pcd, fracture_mask, base_color=(0.6, 0.6, 0.6), fracture_color=(1.0, 0.0, 0.0)):
    colors = np.tile(np.array(base_color, dtype=np.float64), (len(fracture_mask), 1))
    fracture_idx = np.where(fracture_mask > 0.5)[0]
    colors[fracture_idx] = fracture_color
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def main():
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    file_names = ["1776a.obj", "2480.obj"]
    path_a = os.path.join(raw_dir, file_names[0])
    path_b = os.path.join(raw_dir, file_names[1])

    graph_a, graph_b, pcd_a, pcd_b, normalization = preprocess_pair(path_a, path_b)

    for name, graph in zip(file_names, [graph_a, graph_b]):
        save_name = name.replace(".obj", ".pt")
        save_path = os.path.join(processed_dir, save_name)
        torch.save(graph, save_path)

        print(f"Saved: {save_path}")
        print(f"  x: {graph.x.shape}")
        print(f"  Punti frattura: {int(graph.x[:, 7].sum())} / {graph.x.shape[0]}")

    norm_path = os.path.join(processed_dir, "normalization.json")
    with open(norm_path, "w") as f:
        json.dump(normalization, f, indent=2)

    fracture_mask_a = graph_a.x[:, 7].numpy()
    fracture_mask_b = graph_b.x[:, 7].numpy()

    colorize_by_fracture_mask(pcd_a, fracture_mask_a)
    colorize_by_fracture_mask(pcd_b, fracture_mask_b)

    o3d.visualization.draw_geometries(
        [pcd_a, pcd_b],
        window_name="Fracture mask (rosso = frattura stimata)",
    )


if __name__ == "__main__":
    main()
