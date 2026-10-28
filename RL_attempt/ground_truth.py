import torch
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1. MATRICE GROUND TRUTH (CloudCompare)
# -------------------------------------------------------------------------
T_GT = torch.tensor([
    [-0.264350, -0.115371,  0.957501,  32.685013],
    [ 0.196923, -0.978359, -0.063517, -44.966434],
    [ 0.944108,  0.171764,  0.281348,   4.563776],
    [ 0.000000,  0.000000,  0.000000,   1.000000]
], dtype=torch.float32)

def extract_true_fracture_masks(frag_a_pts, frag_b_gt_pts, threshold=0.08):
    with torch.no_grad():
        dists = torch.cdist(frag_a_pts, frag_b_gt_pts)
        min_a2b, _ = torch.min(dists, dim=1)
        min_b2a, _ = torch.min(dists, dim=0)
        return min_a2b < threshold, min_b2a < threshold

import torch
import torch.nn.functional as F

def compute_fracture_metrics(frag_a, frag_b, normals_a, normals_b, mask_a=None, mask_b=None):
    """
    Calcola Chamfer Distance e Normal Alignment garantendo il determinismo al 100%
    tramite l'orientamento forzato dal centro di massa (risolve il Sign Flip SVD/PCA).
    """
    with torch.no_grad():
        # 1. Filtriamo i punti e le normali della superficie di spaccatura
        pts_a_surf = frag_a[mask_a] if (mask_a is not None and torch.any(mask_a)) else frag_a
        norm_a_surf = normals_a[mask_a] if (mask_a is not None and torch.any(mask_a)) else normals_a

        pts_b_surf = frag_b[mask_b] if (mask_b is not None and torch.any(mask_b)) else frag_b
        norm_b_surf = normals_b[mask_b] if (mask_b is not None and torch.any(mask_b)) else normals_b

        # 2. Chamfer Distance
        dists = torch.cdist(pts_b_surf, pts_a_surf)
        min_b2a = torch.min(dists, dim=1).values
        min_a2b = torch.min(dists, dim=0).values
        chamfer_surf = (torch.mean(min_b2a) + torch.mean(min_a2b)).item()

        # 3. Calcolo dei vettori medi di superficie
        mean_norm_a = torch.mean(norm_a_surf, dim=0)
        mean_norm_b = torch.mean(norm_b_surf, dim=0)

        # =========================================================================
        # 🎯 FIX DEFINITIVO: FORZATURA DEL VERSO USCENTE (OUTWARD ORIENTATION)
        # =========================================================================
        # Vettore dal Centro dell'Osso A verso la Superficie di Frattura A
        center_frag_a = torch.mean(frag_a, dim=0)
        center_surf_a = torch.mean(pts_a_surf, dim=0)
        outward_dir_a = center_surf_a - center_frag_a

        # Vettore dal Centro dell'Osso B verso la Superficie di Frattura B
        center_frag_b = torch.mean(frag_b, dim=0)
        center_surf_b = torch.mean(pts_b_surf, dim=0)
        outward_dir_b = center_surf_b - center_frag_b

        # Se la normale media punta VERSO L'INTERNO dell'osso (prodotto scalare negativo), LA INVERTIAMO!
        if torch.dot(mean_norm_a, outward_dir_a) < 0:
            mean_norm_a = -mean_norm_a

        if torch.dot(mean_norm_b, outward_dir_b) < 0:
            mean_norm_b = -mean_norm_b

        # 4. Normalizzazione e Prodotto Scalare
        unit_a = F.normalize(mean_norm_a, p=2, dim=0)
        unit_b = F.normalize(mean_norm_b, p=2, dim=0)

        global_dot = torch.dot(unit_a, unit_b).item()

        # Mappatura [0.0, 1.0]:
        #  - Superfici opposte uscenti (incastro perfetto, dot ≈ -1.0) -> normal_align = 1.0
        #  - Ortogonali (dot ≈ 0.0) -> normal_align = 0.5
        #  - Stessa direzione (dot ≈ +1.0) -> normal_align = 0.0
        normal_align_surf = 0.5 * (1.0 - global_dot)

        dists_all = torch.cdist(frag_b, frag_a)
        min_dist_all_b2a = torch.min(dists_all, dim=1).values
        num_contacts = torch.sum(
                        min_dist_all_b2a < 0.5
                    ).item()

        return chamfer_surf, normal_align_surf, num_contacts
    
# -------------------------------------------------------------------------
# 2. CALCOLO ERRORE DI POSA E REWARD RL (Per lo step dell'Ambiente Gym)
# -------------------------------------------------------------------------
def compute_pose_error(T_agent, T_GT):
    """
    Calcola l'errore di traslazione (euclideo) e di rotazione (in gradi)
    tra la posa corrente dell'agente e la Ground Truth T_GT.
    """
    R_agent, t_agent = T_agent[:3, :3], T_agent[:3, 3]
    R_gt, t_gt = T_GT[:3, :3], T_GT[:3, 3]

    # Errore di traslazione
    trans_error = torch.norm(t_agent - t_gt).item()

    # Errore di rotazione (Geodesic distance su SO(3))
    R_diff = torch.mm(R_agent.t(), R_gt)
    trace = torch.trace(R_diff)
    cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    rot_error_deg = torch.rad2deg(torch.acos(cos_theta)).item()

    return trans_error, rot_error_deg


def compute_rl_reward(T_current, T_GT, prev_trans_err, prev_rot_err, w_trans=1.0, w_rot=0.02):
    """
    Reward basata su Pose Error con segnale di avanzamento denso (Shaped Reward).
    Dà una reward positiva quando l'agente riduce l'errore e una penalità quando lo aumenta.
    """
    trans_err, rot_err = compute_pose_error(T_current, T_GT)

    # Delta di miglioramento ad ogni step
    delta_trans = prev_trans_err - trans_err
    delta_rot = prev_rot_err - rot_err

    # Reward totale densa
    reward = (w_trans * delta_trans) + (w_rot * delta_rot)

    # Bonus di successo se l'agente arriva vicinissimo alla Ground Truth
    is_success = (trans_err < 0.02) and (rot_err < 3.0)
    if is_success:
        reward += 10.0

    return reward, trans_err, rot_err, is_success

def apply_gt_transformation(frag_b_pts, frag_b_normals, T, scale_factor=1.0):
    """
    Applica la rotazione e la traslazione scalata proporzionalmente al dataset.
    """
    R = T[:3, :3]
    # Scaliamo la traslazione in base alla dimensione dell'osso normalizzato
    t = T[:3, 3] / scale_factor
    
    frag_b_pts_trans = torch.mm(frag_b_pts, R.t()) + t
    frag_b_normals_trans = torch.mm(frag_b_normals, R.t())
    return frag_b_pts_trans, frag_b_normals_trans

def plot_single_3d(frag_a, frag_b, chamfer, normal_align):
    pts_a = frag_a.detach().cpu().numpy()[:, :3]
    pts_b = frag_b.detach().cpu().numpy()[:, :3]

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 🚀 Punti ben cicciotti (s=35) per vederli chiaramente
    ax.scatter(pts_a[:, 0], pts_a[:, 1], pts_a[:, 2], c='gold', s=35, label='Frammento A', alpha=0.8)
    ax.scatter(pts_b[:, 0], pts_b[:, 1], pts_b[:, 2], c='deepskyblue', s=35, label='Frammento B (GT)', alpha=0.8)

    # 🚀 Zoom strettissimo attorno alle due ossa
    all_pts = np.vstack([pts_a, pts_b])
    min_x, max_x = all_pts[:, 0].min(), all_pts[:, 0].max()
    min_y, max_y = all_pts[:, 1].min(), all_pts[:, 1].max()
    min_z, max_z = all_pts[:, 2].min(), all_pts[:, 2].max()

    max_range = max(max_x - min_x, max_y - min_y, max_z - min_z) / 2.0
    mid_x = (max_x + min_x) * 0.5
    mid_y = (max_y + min_y) * 0.5
    mid_z = (max_z + min_z) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_title(f"Chamfer: {chamfer:.4f} | Normal Align: {normal_align:.4f}", fontweight='bold', fontsize=12)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def ruota_asse(points, gradi, asse='y'):
    """Ruota i punti di un angolo espresso in gradi attorno all'asse specificato."""
    rad = np.radians(gradi)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    
    if asse == 'x':
        R = torch.tensor([[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]], dtype=torch.float32)
    elif asse == 'y':
        R = torch.tensor([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]], dtype=torch.float32)
    elif asse == 'z':
        R = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=torch.float32)
        
    return torch.mm(points, R.t())


def mask_overlap(mask_static, mask_gt):
    """
    Intersection-over-Union tra la maschera EURISTICA (statica, quella
    salvata in preprocessing e usata dall'ambiente RL) e la maschera
    "vera" derivata qui dalla posa GT nota.

    Un IoU vicino a 0 conferma che l'euristica marca punti diversi da
    quelli davvero a contatto nella frattura reale — a prescindere
    da quanto sia bravo l'agente RL a trovare la posa giusta.
    """
    inter = torch.sum(mask_static & mask_gt).item()
    union = torch.sum(mask_static | mask_gt).item()
    iou = inter / union if union > 0 else 0.0

    return {
        "n_static": int(mask_static.sum().item()),
        "n_gt": int(mask_gt.sum().item()),
        "intersection": inter,
        "union": union,
        "iou": iou,
    }


# -------------------------------------------------------------------------
# 2. ESECUZIONE
# -------------------------------------------------------------------------
if __name__ == "__main__":
    from dataset import BoneFragmentsDataset

    dataset = BoneFragmentsDataset(
        processed_dir="data/processed", num_samples=2000
    )
    frag_a_full, frag_b_full, _ = dataset[0]

    if hasattr(frag_a_full, "x"):
        frag_a_full = frag_a_full.x
    if hasattr(frag_b_full, "x"):
        frag_b_full = frag_b_full.x

    frag_a_pts, frag_a_normals = frag_a_full[:, :3], frag_a_full[:, 3:6]
    frag_b_pts, frag_b_normals = frag_b_full[:, :3], frag_b_full[:, 3:6]

    # 1. TRASFORMAZIONE GROUND TRUTH PURA (Senza offset manuali aggiunti)
    # 1. ROTAZIONE GROUND TRUTH
    R_GT = T_GT[:3, :3]
    frag_b_gt_pts = torch.mm(frag_b_pts, R_GT.t())
    frag_b_gt_normals = torch.mm(frag_b_normals, R_GT.t())

    # 2. ALLINEAMENTO BARICENTRI
    t_correction = frag_a_pts.mean(dim=0) - frag_b_gt_pts.mean(dim=0)
    frag_b_gt_pts = frag_b_gt_pts + t_correction

    # 3. FINE-TUNING POSIZIONE E ROTAZIONE
    offset_xy = torch.tensor([0.17, -0.25, 0.20])
    frag_b_gt_pts = frag_b_gt_pts + offset_xy
    frag_b_gt_pts = ruota_asse(frag_b_gt_pts, gradi=22.0, asse="y")
    # 🆕 FIX: la stessa rotazione di rifinitura va applicata anche
    # alle normali, altrimenti restano disallineate rispetto alla
    # posa finale dei punti (la traslazione non riguarda le normali,
    # ma la rotazione di 22 gradi sì).
    frag_b_gt_normals = ruota_asse(frag_b_gt_normals, gradi=22.0, asse="y")

    # 2. SELEZIONE STRETTA DELLA VERA FRATTURA (Soglia ridotta a 0.04)
    # Rileva solo i punti che sono VERAMENTE a contatto immediato nella GT
    with torch.no_grad():
        dists = torch.cdist(frag_a_pts, frag_b_gt_pts)
        min_a2b, _ = torch.min(dists, dim=1)
        min_b2a, _ = torch.min(dists, dim=0)

        # Distanza minima reale tra i due frammenti
        min_dist_val = min_a2b.min().item()
        
        # 🚀 SOGLIA DINAMICA: Prende i punti vicini al margine di contatto
        # Tolleranza = distanza minima + offset di sicurezza (0.025)
    adaptive_threshold = min_dist_val + 0.025
        
    mask_a = min_a2b < adaptive_threshold
    mask_b = min_b2a < adaptive_threshold

    print(f"🔍 Distanza minima rilevata: {min_dist_val:.4f}")
    print(f"🎯 Soglia adattiva usata : {adaptive_threshold:.4f}")
    print(f"Punti isolati Frattura A : {mask_a.sum().item()} / {len(frag_a_pts)}")
    print(f"Punti isolati Frattura B : {mask_b.sum().item()} / {len(frag_b_pts)}")

    # 2. CALCOLO METRICHE SULLA VERA INTERFACCIA
    chamfer, normal_align, num_contacts = compute_fracture_metrics(
        frag_a_pts, frag_b_gt_pts, 
        frag_a_normals, frag_b_gt_normals, 
        mask_a, mask_b
    )

    print(f"\n✅ Chamfer Distance (Sola Frattura): {chamfer:.4f}")
    print(f"✅ Normal Alignment (Sola Frattura): {normal_align:.4f}\n")
    print(f"✅ Numero contatti: {num_contacts:.4f}\n")
    


    # 🚀 4. GRAFICO 3D ENORME
    plot_single_3d(frag_a_pts, frag_b_gt_pts, chamfer, normal_align)

    # Estraiamo SOLO i 308 punti di spaccatura
    pts_a_surf = frag_a_pts[mask_a].detach().cpu().numpy()
    pts_b_surf = frag_b_gt_pts[mask_b].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Vista X-Y (Dall'alto)
    axes[0].scatter(pts_a_surf[:, 0], pts_a_surf[:, 1], c='gold', s=15, label='Frattura A')
    axes[0].scatter(pts_b_surf[:, 0], pts_b_surf[:, 1], c='deepskyblue', s=15, label='Frattura B')
    axes[0].set_title("Vista X-Y (Dall'alto)")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].grid(True)
    axes[0].legend()

    # 2. Vista X-Z (Frontale)
    axes[1].scatter(pts_a_surf[:, 0], pts_a_surf[:, 2], c='gold', s=15)
    axes[1].scatter(pts_b_surf[:, 0], pts_b_surf[:, 2], c='deepskyblue', s=15)
    axes[1].set_title("Vista X-Z (Frontale)")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Z")
    axes[1].grid(True)

    # 3. Vista Y-Z (Laterale - Profondità)
    axes[2].scatter(pts_a_surf[:, 1], pts_a_surf[:, 2], c='gold', s=15)
    axes[2].scatter(pts_b_surf[:, 1], pts_b_surf[:, 2], c='deepskyblue', s=15)
    axes[2].set_title("Vista Y-Z (Profondità)")
    axes[2].set_xlabel("Y")
    axes[2].set_ylabel("Z")
    axes[2].grid(True)

    plt.suptitle(f"SOLO SUPERFICI DI FRATTURA (308 punti)\nChamfer: {chamfer:.4f} | Normal Align: {normal_align:.4f}", fontweight='bold')
    plt.tight_layout()
    plt.show()

    # =======================================================================
    # 🆕 DIAGNOSI: la maschera STATICA del dataset (quella usata dall'env RL,
    # frag_full[:, 7]) e' affidabile? Qui sopra hai calcolato chamfer=0.045
    # usando pero' una maschera "adattiva" costruita a partire dalla
    # vicinanza stessa alla posa GT (mask_a/mask_b) — non e' la stessa
    # maschera che l'ambiente RL usa durante il training, quindi non e'
    # direttamente confrontabile con il chamfer=22mm del modello.
    #
    # Qui sotto ripetiamo lo stesso calcolo, alla STESSA posa GT rifinita,
    # ma con la maschera STATICA (frag_a_full[:,7] / frag_b_full[:,7]):
    # se anche questa da' un valore alto, la maschera statica e' il vero
    # collo di bottiglia, non l'agente.
    # =======================================================================

    SCALE = 70.0  # stessa scala usata in reset() dell'ambiente RL

    static_mask_a = frag_a_full[:, 7] > 0.5
    static_mask_b = frag_b_full[:, 7] > 0.5

    chamfer_static_norm, normal_align_static, _ = compute_fracture_metrics(
        frag_a_pts, frag_b_gt_pts,
        frag_a_normals, frag_b_gt_normals,
        static_mask_a, static_mask_b
    )

    chamfer_static_mm, _, _ = compute_fracture_metrics(
        frag_a_pts * SCALE, frag_b_gt_pts * SCALE,
        frag_a_normals, frag_b_gt_normals,
        static_mask_a, static_mask_b
    )

    print("=" * 60)
    print("🆕 TEST — Chamfer alla posa GT, con la maschera STATICA")
    print("   (quella realmente usata dall'ambiente RL)")
    print("=" * 60)
    print(f"Chamfer (unita' normalizzate): {chamfer_static_norm:.4f}")
    print(f"Chamfer (mm, scala *{SCALE:.0f}):    {chamfer_static_mm:.4f}")
    print(f"Normal alignment:               {normal_align_static:.4f}")
    print(
        "-> Se chamfer_static_mm e' gia' alto (vicino ai ~22mm del\n"
        "   modello) NONOSTANTE la posa sia GT (quindi perfetta), la\n"
        "   maschera statica e' il vero collo di bottiglia: l'agente\n"
        "   non potra' MAI scendere sotto un certo chamfer, qualunque\n"
        "   posa trovi, perche' la maschera misura i punti sbagliati.\n"
        "   Se invece viene basso, il problema e' altrove (l'agente non\n"
        "   ha davvero trovato la posa corretta, nonostante l'aspetto).\n"
    )

    stats_a = mask_overlap(static_mask_a, mask_a)
    stats_b = mask_overlap(static_mask_b, mask_b)

    print("=" * 60)
    print("🆕 TEST — Overlap maschera statica vs. maschera derivata da GT")
    print("=" * 60)
    print(f"Frammento A: {stats_a}")
    print(f"Frammento B: {stats_b}")
    print(
        "-> IoU vicino a 0 = la maschera statica marca punti quasi\n"
        "   completamente diversi da quelli davvero a contatto nella\n"
        "   frattura reale. IoU alto (es. >0.5) = la maschera e' sensata,\n"
        "   il problema e' probabilmente altrove.\n"
    )

    # =======================================================================
    # 🆕 VISUALIZZAZIONE: dove sbaglia esattamente la maschera statica?
    # Confronto affiancato, sugli stessi frammenti alla posa GT:
    # a sinistra la maschera STATICA (quella usata dall'env RL),
    # a destra la maschera VERA (derivata dalla GT nota).
    # =======================================================================

    pa = frag_a_pts.detach().cpu().numpy()
    pb = frag_b_gt_pts.detach().cpu().numpy()
    sm_a = static_mask_a.detach().cpu().numpy()
    sm_b = static_mask_b.detach().cpu().numpy()
    gm_a = mask_a.detach().cpu().numpy()
    gm_b = mask_b.detach().cpu().numpy()

    fig = plt.figure(figsize=(14, 7))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(pa[~sm_a, 0], pa[~sm_a, 1], pa[~sm_a, 2], c='lightgray', s=3)
    ax1.scatter(pb[~sm_b, 0], pb[~sm_b, 1], pb[~sm_b, 2], c='lightblue', s=3)
    ax1.scatter(pa[sm_a, 0], pa[sm_a, 1], pa[sm_a, 2], c='red', s=12, label='A - maschera statica')
    ax1.scatter(pb[sm_b, 0], pb[sm_b, 1], pb[sm_b, 2], c='darkorange', s=12, label='B - maschera statica')
    ax1.set_title(f"MASCHERA STATICA (env RL)\nChamfer: {chamfer_static_norm:.4f}")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(pa[~gm_a, 0], pa[~gm_a, 1], pa[~gm_a, 2], c='lightgray', s=3)
    ax2.scatter(pb[~gm_b, 0], pb[~gm_b, 1], pb[~gm_b, 2], c='lightblue', s=3)
    ax2.scatter(pa[gm_a, 0], pa[gm_a, 1], pa[gm_a, 2], c='green', s=12, label='A - maschera vera (GT)')
    ax2.scatter(pb[gm_b, 0], pb[gm_b, 1], pb[gm_b, 2], c='limegreen', s=12, label='B - maschera vera (GT)')
    ax2.set_title(f"MASCHERA VERA (da GT)\nChamfer: {chamfer:.4f}")
    ax2.legend(fontsize=8)

    plt.suptitle("Dove marca la frattura la maschera statica, rispetto a dove e' davvero", fontweight='bold')
    plt.tight_layout()
    plt.show()

    # =======================================================================
    # 🆕 GRID SEARCH: il bordo aperto si e' dimostrato inaffidabile su
    # questa mesh (traccia l'intero perimetro visibile, non solo la
    # frattura). Cerchiamo sistematicamente, usando SOLO rugosita'/
    # curvatura (nessun bordo), quale combinazione score+soglia
    # massimizza l'IoU contro la maschera vera (mask_a / mask_b).
    # =======================================================================

    from sklearn.neighbors import KDTree as _KDTree
    from sklearn.cluster import DBSCAN as _DBSCAN

    def compute_roughness_and_curvature_np(points, normals, k=20):
        N = len(points)
        tree = _KDTree(points)
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

    def candidate_mask_from_score(points, score, percentile, eps=0.05, min_samples=4, min_cluster_size=8):
        threshold = np.percentile(score, percentile)
        candidate_idx = np.where(score >= threshold)[0]

        mask = np.zeros(len(points), dtype=bool)
        if len(candidate_idx) == 0:
            return mask

        candidate_pts = points[candidate_idx]
        db = _DBSCAN(eps=eps, min_samples=min_samples).fit(candidate_pts)
        labels = db.labels_

        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        valid = unique_labels[counts >= min_cluster_size]

        if len(valid) == 0:
            return mask

        keep_idx = candidate_idx[np.isin(labels, valid)]
        mask[keep_idx] = True

        return mask

    def iou(mask_pred, mask_gt):
        inter = np.logical_and(mask_pred, mask_gt).sum()
        union = np.logical_or(mask_pred, mask_gt).sum()
        return inter / union if union > 0 else 0.0

    pa_np = frag_a_pts.detach().cpu().numpy()
    na_np = frag_a_normals.detach().cpu().numpy()
    pb_np = frag_b_pts.detach().cpu().numpy()  # frammento B nella SUA posa originale
    nb_np = frag_b_normals.detach().cpu().numpy()

    gm_a_np = mask_a.detach().cpu().numpy()
    gm_b_np = mask_b.detach().cpu().numpy()

    print("Calcolo rugosita'/curvatura per A e B (puo' richiedere qualche secondo)...")
    roughness_a, curvature_a = compute_roughness_and_curvature_np(pa_np, na_np)
    roughness_b, curvature_b = compute_roughness_and_curvature_np(pb_np, nb_np)

    score_variants = {
        "curvatura_pura": (curvature_a, curvature_b),
        "rugosita_pura": (roughness_a, roughness_b),
        "combinata_1x": (roughness_a + curvature_a, roughness_b + curvature_b),
        "combinata_2.5x": (roughness_a + 2.5 * curvature_a, roughness_b + 2.5 * curvature_b),
    }

    percentiles = list(range(80, 99, 2))

    print("\n" + "=" * 70)
    print("🆕 GRID SEARCH — IoU per combinazione score/percentile (media A+B)")
    print("=" * 70)
    print(f"{'score':<16}{'percentile':<12}{'IoU_A':<10}{'IoU_B':<10}{'IoU_media':<10}")

    results = []

    for score_name, (score_a, score_b) in score_variants.items():
        for p in percentiles:
            mask_pred_a = candidate_mask_from_score(pa_np, score_a, p)
            mask_pred_b = candidate_mask_from_score(pb_np, score_b, p)

            iou_a = iou(mask_pred_a, gm_a_np)
            iou_b = iou(mask_pred_b, gm_b_np)
            iou_avg = 0.5 * (iou_a + iou_b)

            results.append((score_name, p, iou_a, iou_b, iou_avg))

    results.sort(key=lambda r: r[4], reverse=True)

    for score_name, p, iou_a, iou_b, iou_avg in results[:15]:
        print(f"{score_name:<16}{p:<12}{iou_a:<10.4f}{iou_b:<10.4f}{iou_avg:<10.4f}")

    best = results[0]
    print(
        f"\n-> Migliore: score='{best[0]}', percentile={best[1]}, IoU medio={best[4]:.4f}\n"
        "   Se anche il migliore resta basso (<0.3), probabilmente serve\n"
        "   un segnale diverso dalla sola curvatura/rugosita' per questo\n"
        "   dataset -- valutiamo insieme il prossimo passo in base a cosa\n"
        "   esce da questa tabella.\n"
    )

    # =======================================================================
    # 🆕 TEST — Chamfer TRONCATO (trimmed): invece della media su TUTTI
    # i punti della maschera statica, usiamo solo la frazione piu'
    # vicina (es. il 20% con distanza minore). Se la maschera contiene
    # comunque una parte di punti veri (IoU non zero, solo diluiti tra
    # falsi positivi), il chamfer troncato dovrebbe avvicinarsi ai
    # punti realmente a contatto e crollare verso valori bassi alla
    # posa GT -- senza bisogno di perfezionare ulteriormente la
    # maschera stessa.
    # =======================================================================

    def trimmed_chamfer(pts_b, pts_a, keep_fraction):
        dists = torch.cdist(pts_b, pts_a)
        min_b2a = torch.min(dists, dim=1).values
        min_a2b = torch.min(dists, dim=0).values

        k_b = max(1, int(len(min_b2a) * keep_fraction))
        k_a = max(1, int(len(min_a2b) * keep_fraction))

        trimmed_b = torch.topk(min_b2a, k_b, largest=False).values.mean()
        trimmed_a = torch.topk(min_a2b, k_a, largest=False).values.mean()

        return (trimmed_b + trimmed_a).item()

    pts_a_static = frag_a_pts[static_mask_a]
    pts_b_static = frag_b_gt_pts[static_mask_b]

    print("=" * 60)
    print("🆕 TEST — Chamfer troncato, maschera statica, posa GT")
    print("=" * 60)
    print(f"{'keep_fraction':<16}{'chamfer (norm)':<18}{'chamfer (mm)':<14}")

    for frac in [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]:
        c_norm = trimmed_chamfer(pts_b_static, pts_a_static, frac)
        c_mm = c_norm * SCALE
        print(f"{frac:<16}{c_norm:<18.4f}{c_mm:<14.2f}")

    print(
        "\n-> Se il chamfer crolla nettamente scendendo con keep_fraction\n"
        "   (es. da ~24mm a poche unita' di mm gia' al 20-10%), la\n"
        "   maschera contiene davvero i punti giusti, solo diluiti --\n"
        "   possiamo integrare un chamfer troncato in\n"
        "   _compute_geometric_metrics() dell'ambiente RL, invece di\n"
        "   continuare a rincorrere una maschera perfetta.\n"
        "   Se invece resta alto anche a keep_fraction bassa, vuol dire\n"
        "   che nemmeno il sottoinsieme piu' vicino della maschera\n"
        "   statica corrisponde alla vera frattura, e serve ripensare\n"
        "   l'estrazione da capo.\n"
    )

    # =======================================================================
    # 🆕 TEST DECISIVO — il chamfer troncato resta ALTO anche a pose
    # SBAGLIATE, o si abbassa facilmente per puro caso geometrico?
    #
    # Il test precedente ha verificato solo che il chamfer troncato
    # riconosca la posa CORRETTA (condizione necessaria, non
    # sufficiente). Qui verifichiamo l'altra meta': che NON dia un
    # falso positivo (valore basso) anche a pose chiaramente sbagliate
    # -- se lo fa, il chamfer troncato e' "barabile" dall'agente RL
    # (reward hacking) e non va usato come reward/metrica di successo.
    # =======================================================================

    def random_rigid_transform(seed):
        g = torch.Generator().manual_seed(seed)
        # Rotazione casuale (angoli Euler ampi, ben lontani da GT)
        angles = torch.rand(3, generator=g) * 360.0
        Rx_ = ruota_asse(torch.eye(3), angles[0].item(), 'x')
        Ry_ = ruota_asse(Rx_, angles[1].item(), 'y')
        R_ = ruota_asse(Ry_, angles[2].item(), 'z')

        # Traslazione casuale moderata (stessa scala normalizzata usata qui)
        t_ = (torch.rand(3, generator=g) - 0.5) * 0.6

        return R_, t_

    print("=" * 60)
    print("🆕 TEST — Chamfer troncato su pose SBAGLIATE (casuali)")
    print("=" * 60)
    print(f"{'seed':<8}{'chamfer_full_mm':<18}{'chamfer_trim20_mm':<20}{'matching_ratio':<16}")

    for wrong_seed in range(5):
        R_wrong, t_wrong = random_rigid_transform(wrong_seed)

        frag_b_wrong_pts = torch.mm(frag_b_pts, R_wrong.t()) + t_wrong
        # Ricentriamo sul baricentro di A, cosi' i frammenti restano
        # almeno nella stessa area (altrimenti sarebbero cosi' lontani
        # che qualsiasi metrica direbbe "sbagliato", e il test non
        # sarebbe interessante: vogliamo pose sbagliate ma PLAUSIBILI,
        # non assurde).
        t_correction_wrong = frag_a_pts.mean(dim=0) - frag_b_wrong_pts.mean(dim=0)
        frag_b_wrong_pts = frag_b_wrong_pts + t_correction_wrong

        pts_b_wrong_static = frag_b_wrong_pts[static_mask_b]

        dists_wrong = torch.cdist(pts_b_wrong_static, pts_a_static)
        min_b2a_wrong = torch.min(dists_wrong, dim=1).values
        min_a2b_wrong = torch.min(dists_wrong, dim=0).values

        chamfer_full_wrong = (torch.mean(min_b2a_wrong) + torch.mean(min_a2b_wrong)).item() * SCALE
        chamfer_trim_wrong = trimmed_chamfer(pts_b_wrong_static, pts_a_static, 0.2) * SCALE

        proximity_band_norm = 2.0 / SCALE
        matching_wrong = 0.5 * (
            (torch.sum(min_b2a_wrong < proximity_band_norm).float() / min_b2a_wrong.numel()).item()
            + (torch.sum(min_a2b_wrong < proximity_band_norm).float() / min_a2b_wrong.numel()).item()
        )

        print(f"{wrong_seed:<8}{chamfer_full_wrong:<18.2f}{chamfer_trim_wrong:<20.2f}{matching_wrong:<16.4f}")

    print(
        "\n-> Se chamfer_trim20_mm risulta basso (es. <15mm, vicino ai\n"
        "   valori che l'agente trova durante il training) ANCHE su\n"
        "   queste pose casuali/sbagliate, e' la prova che il chamfer\n"
        "   troncato non discrimina una posa giusta da una sbagliata --\n"
        "   va abbandonato come reward. chamfer_full_mm invece dovrebbe\n"
        "   restare alto (score non truccabile per costruzione) su pose\n"
        "   sbagliate: e' un confronto diretto tra le due formule.\n"
    )

    # =======================================================================
    # 🆕 TEST — Abbandoniamo la maschera precomputata: matching_ratio
    # calcolato sull'INTERA point cloud (nessun fracture_mask), sia
    # alla posa GT sia sulle stesse pose sbagliate di prima. Se questo
    # discrimina nettamente meglio della versione con maschera, vale
    # la pena ridisegnare l'ambiente RL per lavorare senza maschera
    # precomputata.
    # =======================================================================

    proximity_band_norm = 2.0 / SCALE

    def full_cloud_matching_ratio(pts_b, pts_a, band=proximity_band_norm):
        dists = torch.cdist(pts_b, pts_a)
        min_b2a = torch.min(dists, dim=1).values
        min_a2b = torch.min(dists, dim=0).values

        matching_b = (torch.sum(min_b2a < band).float() / min_b2a.numel()).item()
        matching_a = (torch.sum(min_a2b < band).float() / min_a2b.numel()).item()

        return 0.5 * (matching_b + matching_a)

    print("=" * 60)
    print("🆕 TEST — matching_ratio su TUTTA la point cloud (no maschera)")
    print("=" * 60)

    matching_gt_full = full_cloud_matching_ratio(frag_b_gt_pts, frag_a_pts)
    print(f"Posa GT (corretta):    matching_ratio = {matching_gt_full:.4f}")

    print(f"{'seed':<8}{'matching_ratio (sbagliata)':<28}")
    for wrong_seed in range(5):
        R_wrong, t_wrong = random_rigid_transform(wrong_seed)
        frag_b_wrong_pts = torch.mm(frag_b_pts, R_wrong.t()) + t_wrong
        t_correction_wrong = frag_a_pts.mean(dim=0) - frag_b_wrong_pts.mean(dim=0)
        frag_b_wrong_pts = frag_b_wrong_pts + t_correction_wrong

        matching_wrong_full = full_cloud_matching_ratio(frag_b_wrong_pts, frag_a_pts)
        print(f"{wrong_seed:<8}{matching_wrong_full:<28.4f}")

    print(
        "\n-> Se matching_ratio alla posa GT e' NETTAMENTE piu' alto\n"
        "   delle pose sbagliate (es. GT > 2x il massimo tra le\n"
        "   sbagliate), il segnale sull'intera point cloud discrimina\n"
        "   molto meglio di quello basato sulla maschera -- vale la\n"
        "   pena ridisegnare l'ambiente RL per usarlo, invece di\n"
        "   continuare a inseguire una fracture_mask migliore.\n"
    )

    # =======================================================================
    # 🆕 TEST — "Contatto pulito": vicino E non compenetrante (stesso
    # test del segno della normale gia' usato in _detect_penetration
    # dell'ambiente RL). matching_ratio da solo non distingue "vicino
    # perche' si tocca bene" da "vicino perche' si sta compenetrando"
    # -- qui aggiungiamo quella distinzione.
    # =======================================================================

    def orient_normals_outward(points, normals):
        centroid = points.mean(dim=0, keepdim=True)
        outward_ref = points - centroid
        outward_ref = outward_ref / (torch.norm(outward_ref, dim=1, keepdim=True) + 1e-8)
        dot = torch.sum(normals * outward_ref, dim=1, keepdim=True)
        sign = torch.where(dot < 0, -torch.ones_like(dot), torch.ones_like(dot))
        return normals * sign

    # Orientamento outward calcolato sulla posa ORIGINALE (prima di
    # qualunque trasformazione), poi le normali vengono ruotate
    # insieme ai punti -- stesso ordine di operazioni di environment.py.
    frag_a_normals_oriented = orient_normals_outward(frag_a_pts, frag_a_normals)
    frag_b_normals_oriented = orient_normals_outward(frag_b_pts, frag_b_normals)

    # Normali alla posa GT: stessa sequenza di rotazioni dei punti
    # (R_GT poi i 22 gradi di rifinitura), applicata alle normali
    # gia' orientate outward.
    frag_b_gt_normals_oriented = torch.mm(frag_b_normals_oriented, R_GT.t())
    frag_b_gt_normals_oriented = ruota_asse(frag_b_gt_normals_oriented, gradi=22.0, asse="y")

    def clean_contact_and_penetration(pts_b, pts_a, normals_a, band=proximity_band_norm):
        dists = torch.cdist(pts_b, pts_a)
        min_dist, nearest_idx = torch.min(dists, dim=1)

        nearest_point_a = pts_a[nearest_idx]
        nearest_normal_a = normals_a[nearest_idx]

        vec_a_to_b = pts_b - nearest_point_a
        signed = torch.sum(vec_a_to_b * nearest_normal_a, dim=1)

        close = min_dist < band
        penetrating = (min_dist < band) & (signed < 0)
        clean = close & (signed >= 0)

        return clean.float().mean().item(), penetrating.float().mean().item()

    print("=" * 60)
    print("🆕 TEST — Contatto PULITO (vicino E non compenetrante)")
    print("=" * 60)

    clean_gt, pen_gt = clean_contact_and_penetration(frag_b_gt_pts, frag_a_pts, frag_a_normals_oriented)
    print(f"Posa GT (corretta):    clean_contact = {clean_gt:.4f}   penetration = {pen_gt:.4f}")

    print(f"{'seed':<8}{'clean_contact (sbagliata)':<28}{'penetration (sbagliata)':<26}")
    for wrong_seed in range(5):
        R_wrong, t_wrong = random_rigid_transform(wrong_seed)
        frag_b_wrong_pts = torch.mm(frag_b_pts, R_wrong.t()) + t_wrong
        t_correction_wrong = frag_a_pts.mean(dim=0) - frag_b_wrong_pts.mean(dim=0)
        frag_b_wrong_pts = frag_b_wrong_pts + t_correction_wrong

        clean_wrong, pen_wrong = clean_contact_and_penetration(frag_b_wrong_pts, frag_a_pts, frag_a_normals_oriented)
        print(f"{wrong_seed:<8}{clean_wrong:<28.4f}{pen_wrong:<26.4f}")

    print(
        "\n-> Se penetration alla GT e' vicino a 0 mentre sulle pose\n"
        "   sbagliate e' alto (compenetrazione diffusa dal centrare i\n"
        "   baricentri a caso), e clean_contact alla GT e' comparabile\n"
        "   o superiore alle sbagliate UNA VOLTA ESCLUSA la\n"
        "   compenetrazione, allora la combinazione vicino+non-compenetrante\n"
        "   discrimina bene -- vale la pena ridisegnare l'ambiente RL\n"
        "   su questa base, senza fracture_mask.\n"
    )

    # =======================================================================
    # 🆕 TEST — Stesso confronto, ma su 20 semi casuali invece di 5.
    # 5 pose sbagliate sono poche per fissare una soglia con sicurezza
    # (rischio di ripetere l'errore del chamfer troncato, dove un test
    # troppo piccolo aveva dato un falso senso di sicurezza).
    # =======================================================================

    print("=" * 60)
    print("🆕 TEST — Penetration su 20 semi casuali (validazione soglia)")
    print("=" * 60)

    penetration_wrong_values = []
    for wrong_seed in range(20):
        R_wrong, t_wrong = random_rigid_transform(wrong_seed)
        frag_b_wrong_pts = torch.mm(frag_b_pts, R_wrong.t()) + t_wrong
        t_correction_wrong = frag_a_pts.mean(dim=0) - frag_b_wrong_pts.mean(dim=0)
        frag_b_wrong_pts = frag_b_wrong_pts + t_correction_wrong

        _, pen_wrong = clean_contact_and_penetration(frag_b_wrong_pts, frag_a_pts, frag_a_normals_oriented)
        penetration_wrong_values.append(pen_wrong)

    penetration_wrong_values = np.array(penetration_wrong_values)

    print(f"Posa GT (corretta):        penetration = {pen_gt:.4f}")
    print(f"Pose sbagliate (n=20):     min={penetration_wrong_values.min():.4f}  "
          f"media={penetration_wrong_values.mean():.4f}  "
          f"max={penetration_wrong_values.max():.4f}")
    print(f"Pose sbagliate sotto {pen_gt:.4f} (valore GT): "
          f"{int((penetration_wrong_values <= pen_gt).sum())} / 20")

    print(
        "\n-> Se pochissime (idealmente zero) delle 20 pose sbagliate\n"
        "   scendono sotto il valore della GT, la soglia e' robusta e\n"
        "   possiamo usarla con sicurezza come criterio nell'ambiente RL\n"
        "   (es. penetration_ratio < soglia, calibrata su questo valore).\n"
        "   Se molte pose sbagliate scendono sotto il valore GT, il test\n"
        "   precedente (n=5) era stato un caso fortunato, non un pattern\n"
        "   solido.\n"
    )

    # =======================================================================
    # 🆕 TEST — normal_alignment LOCALE (stessa formula pesata di
    # environment.py: _compute_geometric_metrics), sulla maschera
    # STATICA, GT vs 20 pose sbagliate. Non l'abbiamo ancora validato
    # rigorosamente come gli altri segnali -- verifichiamo se il
    # traguardo di 0.9 nella reward corrisponde davvero a un
    # orientamento corretto, o se e' un altro segnale ingannevole
    # come chamfer e matching_ratio si sono rivelati essere.
    # =======================================================================

    def local_normal_alignment(pts_b, pts_a, norm_b, norm_a, proximity_scale):
        dists = torch.cdist(pts_b, pts_a)
        min_b2a = torch.min(dists, dim=1).values
        idx_b2a = torch.argmin(dists, dim=1)

        corresponding_normal_a = norm_a[idx_b2a]
        local_dot = torch.sum(norm_b * corresponding_normal_a, dim=1)
        local_align = 0.5 * (1.0 - local_dot)

        weight = torch.exp(-min_b2a / proximity_scale)
        weight_sum = weight.sum()

        if weight_sum > 1e-6:
            return (local_align * weight).sum().item() / weight_sum.item()
        return 0.0

    # Stessa collision_threshold dell'ambiente RL (0.5mm), convertita
    # in unita' normalizzate per essere coerente con la scala di questo script.
    collision_threshold_norm = 0.5 / SCALE

    pts_a_static_normals = frag_a_normals_oriented[static_mask_a]
    pts_b_static_normals_gt = frag_b_gt_normals_oriented[static_mask_b]

    normal_align_gt = local_normal_alignment(
        pts_b_static, pts_a_static, pts_b_static_normals_gt, pts_a_static_normals, collision_threshold_norm
    )

    print("=" * 60)
    print("🆕 TEST — normal_alignment LOCALE (maschera statica), GT vs sbagliate")
    print("=" * 60)
    print(f"Posa GT (corretta):    normal_alignment = {normal_align_gt:.4f}")

    print(f"{'seed':<8}{'normal_alignment (sbagliata)':<30}")
    normal_align_wrong_values = []
    for wrong_seed in range(20):
        R_wrong, t_wrong = random_rigid_transform(wrong_seed)
        frag_b_wrong_pts = torch.mm(frag_b_pts, R_wrong.t()) + t_wrong
        t_correction_wrong = frag_a_pts.mean(dim=0) - frag_b_wrong_pts.mean(dim=0)
        frag_b_wrong_pts = frag_b_wrong_pts + t_correction_wrong

        frag_b_wrong_normals = torch.mm(frag_b_normals_oriented, R_wrong.t())

        pts_b_wrong_static = frag_b_wrong_pts[static_mask_b]
        pts_b_wrong_static_normals = frag_b_wrong_normals[static_mask_b]

        na = local_normal_alignment(
            pts_b_wrong_static, pts_a_static, pts_b_wrong_static_normals, pts_a_static_normals, collision_threshold_norm
        )
        normal_align_wrong_values.append(na)
        if wrong_seed < 5:
            print(f"{wrong_seed:<8}{na:<30.4f}")

    normal_align_wrong_values = np.array(normal_align_wrong_values)
    print(f"\nPose sbagliate (n=20): min={normal_align_wrong_values.min():.4f}  "
          f"media={normal_align_wrong_values.mean():.4f}  "
          f"max={normal_align_wrong_values.max():.4f}")
    print(f"Pose sbagliate SOPRA {normal_align_gt:.4f} (valore GT): "
          f"{int((normal_align_wrong_values >= normal_align_gt).sum())} / 20")

    print(
        "\n-> Se molte pose sbagliate arrivano a normal_alignment >= al\n"
        "   valore GT (o addirittura vicino/sopra 0.9), il traguardo\n"
        "   di 0.9 nella reward NON garantisce un orientamento corretto\n"
        "   -- e' un altro segnale ingannevole, e la strategia dei due\n"
        "   stadi (0.9 poi 0.75) andrebbe ripensata sostituendo 0.9 con\n"
        "   un criterio validato (es. penetration_ratio basso) per il\n"
        "   primo stadio.\n"
    )

    # =======================================================================
    # 🆕 TEST — Calibrazione mancante: surface_matching_ratio, STESSA
    # formula esatta di environment.py (maschera statica COMPLETA, non
    # troncata, banda di prossimita' 2mm), alla posa GT vera. Finora
    # SURFACE_MATCHING_TARGET=0.1 non era mai stato validato su GT --
    # stesso errore di calibrazione gia' fatto (e corretto) per
    # normal_alignment (0.9 -> 0.65).
    # =======================================================================

    def surface_matching_ratio_static(min_b2a, min_a2b, band=2.0 / SCALE):
        matching_b = (torch.sum(min_b2a < band).float() / min_b2a.numel()).item()
        matching_a = (torch.sum(min_a2b < band).float() / min_a2b.numel()).item()
        return 0.5 * (matching_b + matching_a)

    dists_static_gt = torch.cdist(pts_b_static, pts_a_static)
    min_b2a_static_gt = torch.min(dists_static_gt, dim=1).values
    min_a2b_static_gt = torch.min(dists_static_gt, dim=0).values

    surface_matching_gt = surface_matching_ratio_static(min_b2a_static_gt, min_a2b_static_gt)

    print("=" * 60)
    print("🆕 TEST — surface_matching_ratio (maschera statica), GT vs sbagliate")
    print("=" * 60)
    print(f"Posa GT (corretta):    surface_matching_ratio = {surface_matching_gt:.4f}")

    surface_matching_wrong_values = []
    for wrong_seed in range(20):
        R_wrong, t_wrong = random_rigid_transform(wrong_seed)
        frag_b_wrong_pts = torch.mm(frag_b_pts, R_wrong.t()) + t_wrong
        t_correction_wrong = frag_a_pts.mean(dim=0) - frag_b_wrong_pts.mean(dim=0)
        frag_b_wrong_pts = frag_b_wrong_pts + t_correction_wrong

        pts_b_wrong_static = frag_b_wrong_pts[static_mask_b]
        dists_wrong_static = torch.cdist(pts_b_wrong_static, pts_a_static)
        min_b2a_wrong_static = torch.min(dists_wrong_static, dim=1).values
        min_a2b_wrong_static = torch.min(dists_wrong_static, dim=0).values

        sm_wrong = surface_matching_ratio_static(min_b2a_wrong_static, min_a2b_wrong_static)
        surface_matching_wrong_values.append(sm_wrong)

    surface_matching_wrong_values = np.array(surface_matching_wrong_values)
    print(f"Pose sbagliate (n=20): min={surface_matching_wrong_values.min():.4f}  "
          f"media={surface_matching_wrong_values.mean():.4f}  "
          f"max={surface_matching_wrong_values.max():.4f}")

    print(
        f"\n-> SURFACE_MATCHING_TARGET attuale = 0.1. Valore vero alla GT = "
        f"{surface_matching_gt:.4f}.\n"
        "   Se il valore vero alla GT e' vicino o sotto 0.1 (come sospettato\n"
        "   guardando il risultato del training), la soglia 0.1 e' troppo\n"
        "   severa -- va ricalibrata verso il basso, come gia' fatto per\n"
        "   normal_alignment (0.9 -> 0.65).\n"
    )

    # =======================================================================
    # 🆕 TEST — La fracture_mask statica si divide in piu' cluster
    # spaziali separati (es. tratto dritto + gancio)? Se si', le
    # metriche aggregate (media pesata, troncata) possono risultare
    # "buone" anche se solo UNO dei cluster tocca davvero l'altro
    # frammento -- il gancio nella foto/render potrebbe essere un
    # cluster separato che la reward attuale ignora.
    # =======================================================================

    from sklearn.cluster import DBSCAN as _DBSCAN2

    def describe_clusters(points_masked, eps=0.05, min_samples=5):
        if len(points_masked) == 0:
            return []
        db = _DBSCAN2(eps=eps, min_samples=min_samples).fit(points_masked.numpy())
        labels = db.labels_
        unique, counts = np.unique(labels[labels != -1], return_counts=True)
        return sorted(counts.tolist(), reverse=True)

    clusters_a = describe_clusters(pts_a_static)
    clusters_b = describe_clusters(pts_b_static)

    print("=" * 60)
    print("🆕 TEST — Cluster spaziali nella fracture_mask statica")
    print("=" * 60)
    print(f"Frammento A: {len(clusters_a)} cluster, dimensioni: {clusters_a}")
    print(f"Frammento B: {len(clusters_b)} cluster, dimensioni: {clusters_b}")
    print(
        "\n-> Se ci sono 2+ cluster di dimensione comparabile (non uno\n"
        "   dominante e uno trascurabile), la maschera rappresenta\n"
        "   davvero piu' zone di contatto separate -- confermerebbe\n"
        "   l'ipotesi: serve richiedere che OGNI cluster tocchi bene,\n"
        "   non solo la media/il sottoinsieme piu' vicino su tutta la\n"
        "   maschera.\n"
    )

    # =======================================================================
    # 🆕 TEST — Tetto vero di surface_matching_ratio A SETTORI (K-Means,
    # K=4, stessa formula di environment.py) alla posa GT, vs 20 pose
    # sbagliate. Fondamentale prima di toccare azioni/osservazioni/reward:
    # se anche a posa corretta il minimo tra i 4 settori resta basso
    # (vicino a quello che il training raggiunge, 0.03), il problema e'
    # il tetto della metrica/maschera, non l'agente o la sua reward.
    # =======================================================================

    from sklearn.cluster import KMeans as _KMeans

    N_SECTORS = 4

    km_a = _KMeans(n_clusters=N_SECTORS, n_init=10, random_state=0).fit(pts_a_static.numpy())
    sector_labels_a = km_a.labels_

    def sector_matching_ratio(min_a2b, sector_labels, band=proximity_band_norm, n_sectors=N_SECTORS):
        ratios = []
        for s in range(n_sectors):
            idx = (sector_labels == s)
            if idx.sum() == 0:
                continue
            ratios.append((min_a2b[idx] < band).float().mean().item())
        return min(ratios) if ratios else 0.0, ratios

    sm_gt_worst, sm_gt_per_sector = sector_matching_ratio(min_a2b_static_gt, sector_labels_a)

    print("=" * 60)
    print("🆕 TEST — surface_matching_ratio A SETTORI (K=4), GT vs sbagliate")
    print("=" * 60)
    print(f"Posa GT (corretta): minimo tra settori = {sm_gt_worst:.4f}  "
          f"(per settore: {[f'{r:.3f}' for r in sm_gt_per_sector]})")

    sm_wrong_worst_values = []
    for wrong_seed in range(20):
        R_wrong, t_wrong = random_rigid_transform(wrong_seed)
        frag_b_wrong_pts = torch.mm(frag_b_pts, R_wrong.t()) + t_wrong
        t_correction_wrong = frag_a_pts.mean(dim=0) - frag_b_wrong_pts.mean(dim=0)
        frag_b_wrong_pts = frag_b_wrong_pts + t_correction_wrong

        pts_b_wrong_static = frag_b_wrong_pts[static_mask_b]
        dists_wrong_static = torch.cdist(pts_b_wrong_static, pts_a_static)
        min_a2b_wrong_static = torch.min(dists_wrong_static, dim=0).values

        sm_wrong, _ = sector_matching_ratio(min_a2b_wrong_static, sector_labels_a)
        sm_wrong_worst_values.append(sm_wrong)

    sm_wrong_worst_values = np.array(sm_wrong_worst_values)
    print(f"Pose sbagliate (n=20): min={sm_wrong_worst_values.min():.4f}  "
          f"media={sm_wrong_worst_values.mean():.4f}  "
          f"max={sm_wrong_worst_values.max():.4f}")

    print(
        f"\n-> Se il valore vero alla GT ({sm_gt_worst:.4f}) e' vicino a "
        f"quello che il training raggiunge (0.03), il tetto e' della\n"
        "   metrica/maschera -- niente da guadagnare cambiando azioni,\n"
        "   osservazioni o reward. Se invece e' molto piu' alto, c'e'\n"
        "   davvero margine, e alcune delle idee di Gemini (con le\n"
        "   correzioni indicate) potrebbero aiutare a raggiungerlo.\n"
    )

    # =======================================================================
    # 🆕 TEST — RICALIBRAZIONE con la formula ESATTA ora in uso in
    # environment.py: matching_a = 0.6*media_settori + 0.4*minimo_settori
    # (non piu' il minimo puro). Il valore GT=0.0698 calcolato prima era
    # con la formula VECCHIA (minimo puro) -- non e' piu' il riferimento
    # corretto dopo il fix del bug che ha attivato il blend.
    # =======================================================================

    def surface_matching_ratio_blend(min_b2a, min_a2b, sector_labels, band=proximity_band_norm, n_sectors=N_SECTORS):
        matching_b = (torch.sum(min_b2a < band).float() / min_b2a.numel()).item()

        ratios = []
        for s in range(n_sectors):
            idx = (sector_labels == s)
            if idx.sum() == 0:
                continue
            ratios.append((min_a2b[idx] < band).float().mean().item())

        if len(ratios) > 0:
            cluster_mean = float(np.mean(ratios))
            cluster_min = float(np.min(ratios))
            matching_a = 0.6 * cluster_mean + 0.4 * cluster_min
        else:
            matching_a = 0.0

        return 0.5 * (matching_b + matching_a)

    sm_blend_gt = surface_matching_ratio_blend(min_b2a_static_gt, min_a2b_static_gt, sector_labels_a)

    print("=" * 60)
    print("🆕 TEST — surface_matching_ratio BLEND (formula esatta in uso), GT vs sbagliate")
    print("=" * 60)
    print(f"Posa GT (corretta): {sm_blend_gt:.4f}")

    sm_blend_wrong_values = []
    for wrong_seed in range(20):
        R_wrong, t_wrong = random_rigid_transform(wrong_seed)
        frag_b_wrong_pts = torch.mm(frag_b_pts, R_wrong.t()) + t_wrong
        t_correction_wrong = frag_a_pts.mean(dim=0) - frag_b_wrong_pts.mean(dim=0)
        frag_b_wrong_pts = frag_b_wrong_pts + t_correction_wrong

        pts_b_wrong_static = frag_b_wrong_pts[static_mask_b]
        dists_wrong_static = torch.cdist(pts_b_wrong_static, pts_a_static)
        min_b2a_wrong_static = torch.min(dists_wrong_static, dim=1).values
        min_a2b_wrong_static = torch.min(dists_wrong_static, dim=0).values

        sm_blend_wrong = surface_matching_ratio_blend(min_b2a_wrong_static, min_a2b_wrong_static, sector_labels_a)
        sm_blend_wrong_values.append(sm_blend_wrong)

    sm_blend_wrong_values = np.array(sm_blend_wrong_values)
    print(f"Pose sbagliate (n=20): min={sm_blend_wrong_values.min():.4f}  "
          f"media={sm_blend_wrong_values.mean():.4f}  "
          f"max={sm_blend_wrong_values.max():.4f}")
    print(f"Pose sbagliate sopra il valore GT ({sm_blend_gt:.4f}): "
          f"{int((sm_blend_wrong_values >= sm_blend_gt).sum())} / 20")

    print(
        f"\n-> Questo e' il vero riferimento da usare ora per SURFACE_MATCHING_TARGET,\n"
        f"   non piu' 0.0698 (quello era calcolato con la formula vecchia,\n"
        f"   minimo puro). Ricalibra la soglia su questo nuovo valore.\n"
    )

    # =======================================================================
    # 🆕 TEST — Sovrapposizione ASSIALE: un vero incastro dovrebbe unire
    # i due frammenti "in punta" (poca sovrapposizione lungo l'asse
    # principale), non farli scorrere paralleli per tutta la lunghezza
    # (molta sovrapposizione). Segnale indipendente dalla fracture_mask
    # -- guarda l'estensione COMPLESSIVA dei due frammenti, non solo la
    # zona marcata come frattura.
    # =======================================================================

    def principal_axis(points):
        centered = points - points.mean(dim=0)
        _, _, V = torch.linalg.svd(centered)
        return V[0]  # sign-invariant per il calcolo di overlap sotto

    def axial_overlap_fraction(pts_a, pts_b, axis, center):
        proj_a = torch.matmul(pts_a - center, axis)
        proj_b = torch.matmul(pts_b - center, axis)

        a_min, a_max = proj_a.min().item(), proj_a.max().item()
        b_min, b_max = proj_b.min().item(), proj_b.max().item()

        overlap = max(0.0, min(a_max, b_max) - max(a_min, b_min))
        a_len = max(a_max - a_min, 1e-6)
        b_len = max(b_max - b_min, 1e-6)

        # Frazione relativa al frammento PIU' CORTO tra i due: quanto
        # del frammento piu' corto e' "doppiamente coperto" lungo
        # l'asse. Un vero incastro end-to-end dovrebbe dare un valore
        # basso; due frammenti che scorrono paralleli danno un valore
        # vicino a 1.
        return overlap / min(a_len, b_len)

    axis_a = principal_axis(frag_a_pts)
    center_a = frag_a_pts.mean(dim=0)

    overlap_gt = axial_overlap_fraction(frag_a_pts, frag_b_gt_pts, axis_a, center_a)

    print("=" * 60)
    print("🆕 TEST — Sovrapposizione assiale (asse principale di A), GT vs sbagliate")
    print("=" * 60)
    print(f"Posa GT (corretta): overlap assiale = {overlap_gt:.4f}")

    overlap_wrong_values = []
    for wrong_seed in range(20):
        R_wrong, t_wrong = random_rigid_transform(wrong_seed)
        frag_b_wrong_pts = torch.mm(frag_b_pts, R_wrong.t()) + t_wrong
        t_correction_wrong = frag_a_pts.mean(dim=0) - frag_b_wrong_pts.mean(dim=0)
        frag_b_wrong_pts = frag_b_wrong_pts + t_correction_wrong

        overlap_wrong = axial_overlap_fraction(frag_a_pts, frag_b_wrong_pts, axis_a, center_a)
        overlap_wrong_values.append(overlap_wrong)

    overlap_wrong_values = np.array(overlap_wrong_values)
    print(f"Pose sbagliate (n=20): min={overlap_wrong_values.min():.4f}  "
          f"media={overlap_wrong_values.mean():.4f}  "
          f"max={overlap_wrong_values.max():.4f}")
    print(f"Pose sbagliate SOTTO il valore GT ({overlap_gt:.4f}): "
          f"{int((overlap_wrong_values <= overlap_gt).sum())} / 20")

    print(
        "\n-> Vogliamo che l'overlap assiale alla GT sia BASSO, e che le\n"
        "   pose sbagliate (specialmente quelle che nidificano due\n"
        "   frammenti simili paralleli, come nel render osservato) diano\n"
        "   valori ALTI. Se la maggior parte delle pose sbagliate ha\n"
        "   overlap maggiore della GT, il segnale discrimina bene e vale\n"
        "   la pena aggiungerlo come penalita' nella reward.\n"
    )

    # =======================================================================
    # 🆕 TEST — Direzione "punta" (vettore baricentro -> punto piu'
    # lontano), un segnale GLOBALE di orientamento senza l'ambiguita'
    # di segno della PCA. A differenza di normal_alignment (locale,
    # guarda solo il punto di contatto), questo cattura il verso
    # complessivo del corpo del frammento -- dovrebbe discriminare
    # "ossa nella stessa direzione" da "ossa in direzioni opposte",
    # indipendentemente da dove si toccano.
    # =======================================================================

    def tip_direction(points):
        centroid = points.mean(dim=0)
        dists = torch.norm(points - centroid, dim=1)
        idx = torch.argmax(dists)
        direction = points[idx] - centroid
        return direction / (torch.norm(direction) + 1e-8)

    tip_dir_a = tip_direction(frag_a_pts)
    tip_dir_b_gt = tip_direction(frag_b_gt_pts)
    tip_alignment_gt = torch.dot(tip_dir_a, tip_dir_b_gt).item()

    print("=" * 60)
    print("🆕 TEST — Direzione punta (baricentro->punto piu' lontano), GT vs sbagliate")
    print("=" * 60)
    print(f"Posa GT (corretta): dot(tip_a, tip_b) = {tip_alignment_gt:.4f}")

    tip_wrong_values = []
    for wrong_seed in range(20):
        R_wrong, t_wrong = random_rigid_transform(wrong_seed)
        frag_b_wrong_pts = torch.mm(frag_b_pts, R_wrong.t()) + t_wrong
        t_correction_wrong = frag_a_pts.mean(dim=0) - frag_b_wrong_pts.mean(dim=0)
        frag_b_wrong_pts = frag_b_wrong_pts + t_correction_wrong

        tip_dir_b_wrong = tip_direction(frag_b_wrong_pts)
        tip_align_wrong = torch.dot(tip_dir_a, tip_dir_b_wrong).item()
        tip_wrong_values.append(tip_align_wrong)

    tip_wrong_values = np.array(tip_wrong_values)
    print(f"Pose sbagliate (n=20): min={tip_wrong_values.min():.4f}  "
          f"media={tip_wrong_values.mean():.4f}  "
          f"max={tip_wrong_values.max():.4f}")
    print(f"Pose sbagliate SOPRA il valore GT ({tip_alignment_gt:.4f}): "
          f"{int((tip_wrong_values >= tip_alignment_gt).sum())} / 20")

    print(
        "\n-> dot() vicino a +1 = punte nella stessa direzione, vicino a\n"
        "   -1 = punte opposte, vicino a 0 = perpendicolari. Se la GT da'\n"
        "   un valore chiaramente diverso (e ripetibile) dalle pose\n"
        "   sbagliate, questo segnale GLOBALE puo' fare da 'polo' -- va\n"
        "   integrato come termine di reward/osservazione separato da\n"
        "   normal_alignment, che resta cieco all'orientamento complessivo.\n"
    )
