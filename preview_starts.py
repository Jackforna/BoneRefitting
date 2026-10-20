"""
Salva le pose di partenza di B (start_mode="cube") per le 6 facce del cubo
attorno ad A, sia come .ply (output_3d/start_<faccia>.ply) sia in
un'immagine riassuntiva (output_3d/start_cube.png), per controllarle
prima di lanciare gli addestramenti.

Uso:
    python preview_starts.py            # roll 0
    python preview_starts.py --roll 90
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from env import ArchaeologicalIncastroEnv
from infer_rl import save_ply
from srcRL.dataset import BoneFragmentsDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--out-dir", default="output_3d")
    args = parser.parse_args()

    env = ArchaeologicalIncastroEnv(BoneFragmentsDataset("data/processed"), start_mode="cube")
    env.START_MAX_TILT = 0.0
    env.START_MAX_AXIAL_SHIFT = 0.0
    os.makedirs(args.out_dir, exist_ok=True)

    faces = list(env.CUBE_FACES)
    fig = plt.figure(figsize=(18, 11))
    for i, face in enumerate(faces):
        env.reset(seed=0, options={"face": face, "roll_deg": args.roll})
        a = env.frag_a.cpu().numpy()
        b = env.frag_b.cpu().numpy()
        save_ply(os.path.join(args.out_dir, f"start_{face}.ply"), a, b)
        _, _, _, min_dist, _, _, pen, _, tip = env._compute_geometric_metrics()
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        ax.scatter(*a[::3].T, s=1, c="gold")
        ax.scatter(*b[::3].T, s=1, c="deepskyblue")
        pts = np.vstack([a, b])
        mid, r = (pts.max(0) + pts.min(0)) / 2, (pts.max(0) - pts.min(0)).max() / 2
        ax.set_xlim(mid[0] - r, mid[0] + r); ax.set_ylim(mid[1] - r, mid[1] + r); ax.set_zlim(mid[2] - r, mid[2] + r)
        ax.set_title(f"{face} | distanza minima {min_dist:.1f} mm | tip {tip:+.2f}")
        print(f"{face:9s}: distanza minima {min_dist:5.1f} mm, penetrazione {100*pen:.2f}%, tip {tip:+.2f}")
    plt.tight_layout()
    path = os.path.join(args.out_dir, "start_cube.png")
    plt.savefig(path, dpi=80)
    print(f"Immagine: {path}  |  .ply: {args.out_dir}/start_<faccia>.ply")


if __name__ == "__main__":
    main()
