"""
Inferenza senza ground truth.

Fa partire il modello PPO addestrato da piu' pose iniziali con le punte
allineate e roll diversi attorno all'asse lungo (start_mode="tip_aligned"
dell'ambiente), segue la policy in modo deterministico e tiene, tra tutte,
la posa con il potenziale Phi piu' alto senza compenetrazione. La posa
scelta viene salvata come .ply (A giallo, B azzurro).

Il roll e' l'unico grado di liberta' che l'agente non riesce a correggere
una volta che le ossa si toccano: provarne diversi e scegliere con Phi
sostituisce la vecchia partenza T_GT senza usare la GT.

Uso (dalla cartella principale del progetto):
    python -m RL_attempt.infer_rl
    python -m RL_attempt.infer_rl --processed examples/data1/processed --confronta-gt
    python -m RL_attempt.infer_rl --model RL_attempt/checkpoints/ppo_incastro_200000_steps.zip \
                                  --vecnorm RL_attempt/checkpoints/ppo_incastro_vecnormalize_200000_steps.pkl

--confronta-gt stampa SOLO A FINE ESECUZIONE la distanza dalla GT
(examples/dataN/gt_B.npy): non influenza in alcun modo la scelta.
"""
import argparse
import os

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env import ArchaeologicalIncastroEnv
from src.dataset import BoneFragmentsDataset


def save_ply(path, pts_a, pts_b):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts_a) + len(pts_b)}\n")
        f.write("property double x\nproperty double y\nproperty double z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p in pts_a:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 255 191 0\n")
        for p in pts_b:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 0 166 255\n")


def current_state(env):
    c, _, cov, min_dist, _, na, pen, _, tip = env._compute_geometric_metrics()
    return {
        "phi": env._potential(c, na, cov, tip),
        "chamfer": c, "surface": cov, "normal": na, "tip": tip,
        "penetration": pen, "min_dist": min_dist,
    }


def run_from_roll(model, vecnorm, env, roll_deg, max_steps, deterministic=True, face=None, rank=None):
    if rank is not None:              # start_mode="searched": posa di search_start.py
        options = {"rank": rank}
    else:
        options = {"roll_deg": roll_deg}
    if face is not None:
        options["face"] = face
    obs, _ = env.reset(options=options)
    best, best_pose = None, None
    for _ in range(max_steps):
        norm_obs = vecnorm.normalize_obs(obs) if vecnorm is not None else obs
        action, _ = model.predict(norm_obs, deterministic=deterministic)
        obs, _, terminated, truncated, _ = env.step(action)
        with torch.no_grad():
            state = current_state(env)
        if state["penetration"] <= env.PENETRATION_RATIO_TARGET and (best is None or state["phi"] > best["phi"]):
            best, best_pose = state, env.frag_b.detach().cpu().numpy().copy()
        if terminated or truncated:
            break
    return best, best_pose


def load_ppo(path, device):
    """
    Carica un modello PPO. I checkpoint in RL_attempt/checkpoints/ sono stati
    salvati quando l'encoder stava in srcRL/InterfaceEncoder.py e quel nome e'
    scritto dentro il file: lo si fa puntare a RL_attempt/InterfaceEncoder.py.
    """
    import sys
    import types
    import RL_attempt.InterfaceEncoder as interface_encoder
    sys.modules.setdefault("srcRL", types.ModuleType("srcRL"))
    sys.modules.setdefault("srcRL.InterfaceEncoder", interface_encoder)
    return PPO.load(path, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="RL_attempt/checkpoints/ppo_incastro_300000_steps.zip")
    parser.add_argument("--vecnorm", default="RL_attempt/checkpoints/ppo_incastro_vecnormalize_300000_steps.pkl")
    parser.add_argument("--roll-step", type=float, default=45.0,
                        help="passo in gradi tra i roll provati (45 = gli stessi roll dell'addestramento)")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--rolls", default=None,
                        help="elenco di roll separati da virgola (es. 270 oppure 0,45,90); sostituisce --roll-step")
    parser.add_argument("--out", default="results/infer_best.ply")
    parser.add_argument("--confronta-gt", action="store_true", help="stampa la distanza dalla GT a fine esecuzione")
    parser.add_argument("--episodi-per-roll", type=int, default=1,
                        help="episodi per ogni roll; con piu' di 1 la policy e' stocastica (come in training) "
                             "e per ogni roll si tiene il Phi migliore")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--searched", action="store_true",
                        help="parte dalle pose di search_start.py invece che dai roll")
    parser.add_argument("--ranks", default="0",
                        help="con --searched: pose della classifica da provare, separate da virgola (es. 0,1,2)")
    parser.add_argument("--start-poses", default="results/runs1/start_search/start_poses.npz")
    parser.add_argument("--processed", default="examples/data1/processed",
                        help="cartella dei .pt (e di occupancy/) della coppia di ossa")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = BoneFragmentsDataset(processed_dir=args.processed)
    model = load_ppo(args.model, device)

    # L'osservazione deve essere quella con cui il modello e' stato addestrato
    obs_mode = "interface" if hasattr(model.observation_space, "spaces") else "geom"
    start_poses = None
    if args.searched:
        from RL_attempt.trainRL import load_start_poses
        start_poses = load_start_poses(args.start_poses)
    env = ArchaeologicalIncastroEnv(
        dataset=dataset, max_steps=args.max_steps, obs_mode=obs_mode,
        n_iface_points=128, device=device, start_mode="searched" if args.searched else "tip_aligned",
        start_poses=start_poses,
    )
    env.START_MAX_TILT = 0.0          # partenze deterministiche in inferenza
    env.START_MAX_AXIAL_SHIFT = 0.0
    env.reset(seed=args.seed)
    torch.manual_seed(args.seed)
    stochastic = args.episodi_per_roll > 1

    vecnorm = None
    if args.vecnorm and os.path.exists(args.vecnorm):
        vecnorm = VecNormalize.load(args.vecnorm, DummyVecEnv([lambda: env]))
        vecnorm.training = False
        vecnorm.norm_reward = False
    else:
        print(f"Statistiche VecNormalize non trovate ({args.vecnorm}): osservazioni non normalizzate.")

    results = []
    print(f"Modello: {args.model} ({obs_mode})  |  "
          f"{'pose di search_start.py: ' + args.ranks if args.searched else 'roll provati: ogni ' + format(args.roll_step, 'g') + ' gradi'}  |  "
          f"{'policy stocastica, ' + str(args.episodi_per_roll) + ' episodi per roll' if stochastic else 'policy deterministica'}")
    if args.searched:
        starts = [("rank", int(r)) for r in args.ranks.split(",")]
    else:
        rolls = ([float(r) for r in args.rolls.split(",")] if args.rolls
                 else list(np.arange(0.0, 360.0, args.roll_step)))
        starts = [("roll", float(r)) for r in rolls]
    for kind, value in starts:
        roll = value
        state, pose = None, None
        for _ in range(args.episodi_per_roll):
            if kind == "rank":
                s_ep, p_ep = run_from_roll(model, vecnorm, env, None, args.max_steps,
                                           deterministic=not stochastic, rank=value)
            else:
                s_ep, p_ep = run_from_roll(model, vecnorm, env, value, args.max_steps, deterministic=not stochastic)
            if s_ep is not None and (state is None or s_ep["phi"] > state["phi"]):
                state, pose = s_ep, p_ep
        if state is None:
            print(f"  {kind} {roll:5g}: nessuna posa senza compenetrazione")
            continue
        results.append((state, pose, roll))
        print(f"  {kind} {roll:5g}: Phi={state['phi']:6.2f} chamfer={state['chamfer']:5.2f} "
              f"surface={state['surface']:.3f} normal={state['normal']:.2f} pen={100*state['penetration']:.2f}%")

    if not results:
        print("Nessun risultato valido.")
        return
    state, pose, roll = max(results, key=lambda r: r[0]["phi"])
    save_ply(args.out, env.frag_a.detach().cpu().numpy(), pose)
    print(f"\n🏆 Scelta (Phi massimo): {'rank' if args.searched else 'roll'} {roll:g} | Phi={state['phi']:.2f} chamfer={state['chamfer']:.2f} mm "
          f"surface={state['surface']:.3f} normal={state['normal']:.2f}")
    print(f"   Salvata in {args.out}")

    if args.confronta_gt:
        from search_start import _carica_gt, _distanza_gt
        gt, gt_path = _carica_gt(args.processed)
        if gt is None:
            print(f"   [verifica] GT non trovata ({gt_path})")
        else:
            media, centroidi, angolo = _distanza_gt(pose, gt)
            print(f"   [verifica] distanza dalla GT: media {media:.1f} mm, centroidi {centroidi:.1f} mm, "
                  f"angolo {angolo:.1f} gradi")


if __name__ == "__main__":
    main()
