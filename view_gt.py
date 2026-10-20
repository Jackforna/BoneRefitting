"""
Visualizzazione 3D (Open3D) della soluzione vera di un esempio esportato con
fractura_export.py (data/gt/<tag>.json), da usare SOLO dopo aver giudicato
rank0.

A giallo, B nella posa vera azzurro (punti di frattura piu' scuri, come in
view_pose.py). Con --runs si sovrappone anche una posa della ricerca, in
verde, per vedere dove si discosta dalla soluzione.

Uso:
    python view_gt.py --gt data/gt/osso1.json --processed data2/processed
    python view_gt.py --gt data/gt/osso1.json --processed data2/processed --runs runs2 --rank 0
"""
import argparse
import json
import os

import numpy as np
import open3d as o3d

from srcRL.dataset import BoneFragmentsDataset
from view_pose import BLUE, CYAN, ORANGE, YELLOW, cloud, colors_for

SCALE = 70.0                      # come env.py
GREEN, DARK_GREEN = [0.2, 0.85, 0.3], [0.0, 0.45, 0.1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="file .json della soluzione (fractura_export.py)")
    parser.add_argument("--processed", required=True, help="cartella dei .pt dello stesso esempio")
    parser.add_argument("--runs", default=None, help="facoltativo: cartella dei risultati da sovrapporre")
    parser.add_argument("--rank", type=int, default=0, help="con --runs: quale posa della classifica")
    parser.add_argument("--punti", type=float, default=3.0, help="dimensione dei punti")
    args = parser.parse_args()

    with open(args.gt) as f:
        gt = json.load(f)
    R, c, t = np.array(gt["R"]), np.array(gt["c"]), np.array(gt["t"])
    with open(os.path.join(args.processed, "normalization.json")) as f:
        norm = json.load(f)
    centro, raggio = np.array(norm["centroid"]), norm["radius"]

    dataset = BoneFragmentsDataset(processed_dir=args.processed)
    a_full, b_full, _ = dataset[0]
    a = (a_full.x if hasattr(a_full, "x") else a_full).numpy()
    b = (b_full.x if hasattr(b_full, "x") else b_full).numpy()
    pts_a, b_rest = a[:, :3] * SCALE, b[:, :3] * SCALE
    mask_a = a[:, 7] > 0.5 if a.shape[1] > 7 else np.zeros(len(a), dtype=bool)
    mask_b = b[:, 7] > 0.5 if b.shape[1] > 7 else np.zeros(len(b), dtype=bool)

    # B esportato = (B_vero - c) @ R.T + c + t, in coordinate .obj -> si torna alla posa vera
    b_obj = b_rest / SCALE * raggio + centro
    b_vero = ((b_obj - c - t) @ R + c - centro) / raggio * SCALE

    geometries = [cloud(pts_a, colors_for(mask_a, YELLOW, ORANGE)),
                  cloud(b_vero, colors_for(mask_b, CYAN, BLUE))]
    title = f"Soluzione vera - {gt['nome']}"
    print(f"{' + '.join(dataset.pairs[0])} | soluzione vera di {gt['nome']} (pezzi {gt['pezzi']})")

    if args.runs:
        data = np.load(os.path.join(args.runs, "start_search", "start_poses.npz"))
        posa = b_rest @ data["R"][args.rank].T + data["t"][args.rank]
        errore = np.linalg.norm(posa - b_vero, axis=1).mean()
        geometries.append(cloud(posa, colors_for(mask_b, GREEN, DARK_GREEN)))
        title += f" + rank {args.rank} (verde)"
        print(f"rank {args.rank} in verde: Phi={data['phi'][args.rank]:.2f}, distanza media dalla soluzione {errore:.1f}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"BoneRefitting - {title}", width=1280, height=900)
    for g in geometries:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = args.punti
    opt.background_color = np.array([0.1, 0.1, 0.1])
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
