"""
Visualizzazione 3D (Open3D) delle pose trovate da search_start.py, o di un
qualsiasi .ply salvato dal progetto.

A giallo, B azzurro; con le maschere di frattura (default) i punti di
frattura sono piu' scuri: arancione su A, blu su B. Nella finestra:
trascina per ruotare, rotella per lo zoom, +/- per la dimensione dei
punti, Q per chiudere.

Uso:
    python view_pose.py                      # rank 0 di runs/start_search/start_poses.npz
    python view_pose.py --rank 1
    python view_pose.py --rank 0 --staccata  # come parte l'RL: B staccato di 15 mm
    python view_pose.py --file output_3d/infer_best.ply
"""
import argparse
import os

import numpy as np
import open3d as o3d

from srcRL.dataset import BoneFragmentsDataset

YELLOW, ORANGE = [1.0, 0.75, 0.0], [0.85, 0.35, 0.0]
CYAN, BLUE = [0.0, 0.65, 1.0], [0.0, 0.2, 0.75]
SCALE = 70.0             # come env.py
START_DISTANCE = 15.0    # mm, distacco della partenza --searched di trainRL.py


def cloud(points, colors):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pc.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return pc


def colors_for(mask, base, fracture):
    colors = np.tile(base, (len(mask), 1))
    colors[mask] = fracture
    return colors


def pose_clouds(poses_path, rank, processed_dir="data5/processed"):
    data = np.load(poses_path)
    if rank >= len(data["phi"]):
        raise SystemExit(f"rank {rank} non presente: il file contiene {len(data['phi'])} pose")
    R, t = data["R"][rank], data["t"][rank]

    dataset = BoneFragmentsDataset(processed_dir=processed_dir) #cambia numero data per vedere un altro esempio d'incastro
    a_full, b_full, _ = dataset[0]
    a = (a_full.x if hasattr(a_full, "x") else a_full).numpy()
    b = (b_full.x if hasattr(b_full, "x") else b_full).numpy()
    pts_a, pts_b = a[:, :3] * SCALE, b[:, :3] * SCALE
    mask_a = a[:, 7] > 0.5 if a.shape[1] > 7 else np.zeros(len(a), dtype=bool)
    mask_b = b[:, 7] > 0.5 if b.shape[1] > 7 else np.zeros(len(b), dtype=bool)

    pts_b = pts_b @ R.T + t

    extra = ""
    if "roll" in data.files:
        extra = f", roll {data['roll'][rank]:g}, scorrimento {data['offset'][rank]:+g} mm"
    print(f"rank {rank}: Phi={data['phi'][rank]:.2f}{extra}")
    return [cloud(pts_a, colors_for(mask_a, YELLOW, ORANGE)),
            cloud(pts_b, colors_for(mask_b, CYAN, BLUE))], f"rank {rank}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0, help="posa della classifica di search_start.py (0 = Phi migliore)")
    parser.add_argument("--runs", default="1", help="cartella dell'esempio (1, 2, 3, 4)")
    parser.add_argument("--processed", default=None, help="solo per file delle pose vecchi, senza le ossa dentro")
    args = parser.parse_args()


    poses = os.path.join(f"runs{args.runs}", "start_search", "start_poses.npz")
    if args.processed == None:
        processed = os.path.join(f"data{args.runs}", "processed")
    else:
        processed = os.path.join(f"data{args.processed}", "processed")
    if not os.path.exists(poses):
        raise SystemExit(f"{args.poses} non trovato: lancia prima python search_start.py")
    geometries, title = pose_clouds(poses, args.rank, processed)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"BoneRefitting - {title}", width=1280, height=900)
    for g in geometries:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = 3.0
    opt.background_color = np.array([0.1, 0.1, 0.1])
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
