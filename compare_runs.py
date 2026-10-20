"""
Confronto tra run addestrati ciascuno da un roll di partenza fisso
(python trainRL.py --roll R ...), per l'eliminazione progressiva.

Per ogni cartella runs/roll_<R>/, runs/face_<F>[_roll_<R>]/ o runs/searched_rank<N>/ prende il modello piu' recente (finale
o ultimo checkpoint), lo fa partire dal SUO roll per alcuni episodi con
la policy stocastica e registra il Phi massimo raggiunto senza
compenetrazione. I run vengono ordinati per Phi: nessuna GT nella scelta.

Uso:
    python compare_runs.py
    python compare_runs.py --episodi 10 --tieni 4
    python compare_runs.py --confronta-gt     # solo verifica, a posteriori
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env import ArchaeologicalIncastroEnv
from infer_rl import gt_distance, run_from_roll, save_ply
from srcRL.dataset import BoneFragmentsDataset
from trainRL import load_start_poses


def latest_model(run_dir):
    """(modello, vecnormalize, step) del run: il finale se c'e', altrimenti l'ultimo checkpoint."""
    final = os.path.join(run_dir, "ppo_archaeological_aligner.zip")
    if os.path.exists(final):
        return final, os.path.join(run_dir, "ppo_archaeological_aligner_vecnormalize.pkl"), "modello finale"
    ckpts = glob.glob(os.path.join(run_dir, "checkpoints", "ppo_incastro_*_steps.zip"))
    if not ckpts:
        return None, None, None
    steps = max(int(re.search(r"_(\d+)_steps\.zip$", c).group(1)) for c in ckpts)
    model = os.path.join(run_dir, "checkpoints", f"ppo_incastro_{steps}_steps.zip")
    vecnorm = os.path.join(run_dir, "checkpoints", f"ppo_incastro_vecnormalize_{steps}_steps.pkl")
    return model, vecnorm, f"checkpoint {steps} step"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodi", type=int, default=5, help="episodi stocastici per run")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--tieni", type=int, default=None, help="quanti run indicare come promossi al turno dopo")
    parser.add_argument("--confronta-gt", action="store_true", help="stampa la distanza dalla GT (solo verifica)")
    args = parser.parse_args()

    runs = sorted(d for d in glob.glob(os.path.join("runs", "*"))
                  if os.path.isdir(d) and os.path.basename(d) != "start_search")
    if not runs:
        print("Nessun run trovato in runs/. Lancia prima: python trainRL.py --face F --steps N --no-viewer")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = BoneFragmentsDataset(processed_dir="data/processed")
    torch.manual_seed(0)
    results = []

    for run_dir in runs:
        config_path = os.path.join(run_dir, "run_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
        else:  # run per roll creati prima di run_config.json
            config = {"start_mode": "tip_aligned", "face": None, "roll": float(run_dir.split("roll_")[-1])}
        searched = config["start_mode"] == "searched"
        roll = None if searched else float(config["roll"])
        face = config.get("face")
        label = os.path.basename(run_dir)
        model_path, vecnorm_path, steps = latest_model(run_dir)
        if model_path is None:
            print(f"  {run_dir}: nessun modello o checkpoint, saltato")
            continue
        model = PPO.load(model_path, device=device)
        obs_mode = "interface" if hasattr(model.observation_space, "spaces") else "geom"
        start_poses = load_start_poses(config["start_poses"]) if searched else None
        env = ArchaeologicalIncastroEnv(dataset=dataset, max_steps=args.max_steps, obs_mode=obs_mode,
                                        n_iface_points=128, device=device, start_mode=config["start_mode"],
                                        start_face=face, start_poses=start_poses)
        env.START_MAX_TILT = 0.0
        env.START_MAX_AXIAL_SHIFT = 0.0
        env.reset(seed=0)
        vecnorm = None
        if os.path.exists(vecnorm_path):
            vecnorm = VecNormalize.load(vecnorm_path, DummyVecEnv([lambda: env]))
            vecnorm.training = False
            vecnorm.norm_reward = False

        # Run con piu' roll (--all-rolls): si provano tutti e si tiene il migliore
        rolls = [None] if searched else [float(r) for r in config.get("rolls") or [roll]]
        rank = config.get("rank", 0) if searched else None
        best, best_pose, best_roll = None, None, None
        for r in rolls:
            for _ in range(args.episodi):
                state, pose = run_from_roll(model, vecnorm, env, r, args.max_steps, deterministic=False,
                                            face=face, rank=rank)
                if state is not None and (best is None or state["phi"] > best["phi"]):
                    best, best_pose, best_roll = state, pose, r
        if len(rolls) > 1:
            label = f"{os.path.basename(run_dir)} (roll migliore {best_roll:g})" if best is not None else label
        if best is None:
            print(f"  {label} ({steps}): nessuna posa senza compenetrazione")
            continue
        save_ply(os.path.join(run_dir, "output_3d", "compare_best.ply"), env.frag_a.detach().cpu().numpy(), best_pose)
        results.append((best["phi"], label, steps, best, best_pose, env))
        print(f"  {label} ({steps}): Phi={best['phi']:6.2f} chamfer={best['chamfer']:5.2f} "
              f"surface={best['surface']:.3f} normal={best['normal']:.2f}", flush=True)

    results.sort(key=lambda r: r[0], reverse=True)
    print("\nClassifica per Phi (nessuna GT nella scelta):")
    for rank, (phi, label, steps, best, pose, env) in enumerate(results, 1):
        mark = "  <- promosso" if args.tieni and rank <= args.tieni else ""
        print(f"  {rank}. {label}: Phi={phi:6.2f} chamfer={best['chamfer']:5.2f} surface={best['surface']:.3f}{mark}")

    if args.confronta_gt and results:
        print("\n[verifica a posteriori] distanza dalla GT calibrata:")
        for phi, label, steps, best, pose, env in results:
            err, angle = gt_distance(env, dataset, pose)
            print(f"  {label}: {err:5.1f} mm, {angle:5.1f} gradi")


if __name__ == "__main__":
    main()
