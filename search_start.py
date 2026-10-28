"""
Ricerca della posa di partenza di B, senza ground truth.

B viene appoggiato con la sua maschera di frattura sopra quella di A
(punte allineate), per ogni combinazione di roll attorno all'asse della
punta e di scorrimento lungo l'asse lungo di A. Da ciascuna partenza una
salita casuale (hill-climb) di piccoli spostamenti e rotazioni accetta
solo le pose senza compenetrazione che aumentano il potenziale Phi
dell'ambiente. Vince la posa con il Phi piu' alto.

La posa vincente (e le successive in classifica) viene salvata come
rototraslazione rispetto alla posa del dataset in
<out>/start_search/start_poses.npz (default runs/start_search/): trainRL.py
--searched la usa come partenza, come una volta si usava T_GT. Nel file
vengono salvati anche i punti e le maschere delle due ossa, cosi'
view_pose.py --runs <out> mostra le pose di ogni esempio anche dopo che
data/processed contiene altre ossa. Nessuna GT nella scelta:
--confronta-gt stampa la distanza dalla GT solo a fine esecuzione.

Uso:
    python search_start.py
    python search_start.py --workers 4 --iters 2500 --roll-step 15
    python search_start.py --processed examples/data1/processed --out results/runs1
"""
import argparse
import math
import multiprocessing as mp
import os
import time
import warnings

import numpy as np
import torch

_env = None
_n0 = None
_w_ext = 0.0
EXT_CONTACT_DIST = 2.0   # mm (unita' dell'ambiente): punti "a contatto" per il termine sulle superfici esterne


def _init_worker(processed_dir="examples/data1/processed", w_tip=None, pen_max=None, w_ext=0.0):
    """Un ambiente per processo (le ricerche girano in parallelo)."""
    global _env, _n0, _w_ext
    _w_ext = w_ext
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    from env import ArchaeologicalIncastroEnv
    from src.dataset import BoneFragmentsDataset

    dataset = BoneFragmentsDataset(processed_dir=processed_dir)
    _env = ArchaeologicalIncastroEnv(dataset=dataset, device="cpu", start_mode="tip_aligned",
                                     occupancy_dir=os.path.join(processed_dir, "occupancy"))
    if w_tip is not None:
        _env.W_TIP = w_tip      # 0 = nessun premio sulle punte allineate (--punte entrambe)
    if pen_max is not None:
        _env.PENETRATION_RATIO_TARGET = pen_max   # compenetrazione massima accettata
    _env.START_MAX_TILT = 0.0
    _env.START_MAX_AXIAL_SHIFT = 0.0
    _env.reset(seed=0)
    _, b_full, _ = dataset[0]
    b_full = b_full.x if hasattr(b_full, "x") else b_full
    _n0 = _env._orient_normals_outward(_env.frag_b_rest, b_full[:, 3:6].float().clone())


def _rot(axis, angle):
    axis = axis / (axis.norm() + 1e-8)
    K = torch.tensor([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return torch.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def _set_pose(pb):
    _env.frag_b = pb
    R, _ = _env._current_pose_b()
    _env.frag_b_normals = _n0 @ R.T


def _contatto_esterno():
    """
    Frazione di punti FUORI dalle maschere di frattura (superficie esterna
    dell'osso) a meno di EXT_CONTACT_DIST dall'altro frammento, sommata su A
    e B. In un incastro vero si toccano solo le fratture.
    """
    d = torch.cdist(_env.frag_b, _env.frag_a)
    vicini_b = d.min(dim=1).values < EXT_CONTACT_DIST
    vicini_a = d.min(dim=0).values < EXT_CONTACT_DIST
    fuori_a, fuori_b = ~_env.fracture_mask_a.bool(), ~_env.fracture_mask_b.bool()
    return float((vicini_b & fuori_b).float().mean() + (vicini_a & fuori_a).float().mean())


def _phi(pb):
    _set_pose(pb)
    c, _, cov, _, _, na, pen, _, tip = _env._compute_geometric_metrics()
    phi = _env._potential(c, na, cov, tip)
    metrics = {"chamfer": c, "surface": cov, "normal": na, "tip": tip}
    if _w_ext:
        est = _contatto_esterno()
        phi -= _w_ext * est
        metrics["esterno"] = est
    return phi, metrics


def _valid(pb):
    _set_pose(pb)
    return _env._detect_penetration()[2] <= _env.PENETRATION_RATIO_TARGET


def _initial_pose(roll_deg, offset_mm, sense=1.0, tilt_deg=0.0, azim_deg=0.0):
    """
    Punta di B verso la punta di A (sense=+1) o verso il lato opposto (-1),
    inclinata di tilt_deg in direzione azim_deg attorno all'asse di A, roll
    dato, maschera di B sulla maschera di A (+ scorrimento assiale).
    """
    env = _env
    b0 = env.frag_b_rest.clone()
    tip_a = sense * env.tip_dir_a / env.tip_dir_a.norm()
    if tilt_deg:
        ref = torch.tensor([1.0, 0.0, 0.0]) if abs(float(tip_a[0])) < 0.9 else torch.tensor([0.0, 1.0, 0.0])
        perp = torch.linalg.cross(tip_a, ref)
        perp = _rot(tip_a, math.radians(azim_deg)) @ (perp / perp.norm())
        tip_a = _rot(perp, math.radians(tilt_deg)) @ tip_a
    tip_b = env._tip_direction(b0)
    cos = float(torch.clamp(torch.dot(tip_b, tip_a), -1.0, 1.0))
    axis = torch.linalg.cross(tip_b, tip_a)
    if axis.norm() < 1e-6:
        axis = torch.linalg.cross(tip_a, torch.tensor([1.0, 0.0, 0.0]))
    R = _rot(tip_a, math.radians(roll_deg)) @ _rot(axis, math.acos(cos))
    pb = (b0 - b0.mean(0)) @ R.T
    mask_a, mask_b = env.fracture_mask_a.bool(), env.fracture_mask_b.bool()
    pb = pb + env.frag_a[mask_a].mean(0) - pb[mask_b].mean(0) + env.principal_axis_a * offset_mm
    # fuori dalla compenetrazione, allontanandosi dal baricentro di A
    for _ in range(300):
        if _valid(pb):
            break
        away = pb.mean(0) - env.frag_a.mean(0)
        pb = pb + away / away.norm()
    return pb


def _climb(task):
    roll_deg, offset_mm, iters, seed, sense, tilt_deg, azim_deg = task
    t0 = time.time()
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    pb = _initial_pose(roll_deg, offset_mm, sense, tilt_deg, azim_deg)
    phi, metrics = _phi(pb)
    for i in range(iters):
        step = 3.0 * (1 - i / iters) + 0.2      # passi grandi all'inizio, fini alla fine
        R = _rot(torch.randn(3, generator=gen), math.radians(rng.normal(0.0, 2.0 * step)))
        center = pb.mean(0)
        cand = (pb - center) @ R.T + center + torch.randn(3, generator=gen) * step
        if not _valid(cand):
            continue
        phi2, metrics2 = _phi(cand)
        if phi2 > phi:
            pb, phi, metrics = cand, phi2, metrics2
    _set_pose(pb)
    R, t = _env._current_pose_b()
    return {"roll": roll_deg, "offset": offset_mm, "sense": sense, "tilt": tilt_deg, "azim": azim_deg, "phi": float(phi), "metrics": metrics,
            "R": R.numpy(), "t": t.numpy(), "pose": pb.numpy(), "seconds": time.time() - t0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roll-step", type=float, default=15.0, help="passo in gradi tra i roll provati")
    parser.add_argument("--offsets", default="-12,0,12", help="scorrimenti lungo l'asse di A (mm)")
    parser.add_argument("--iters", type=int, default=2500, help="iterazioni di hill-climb per partenza")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--punte", choices=["entrambe", "allineate", "opposte"], default="entrambe",
                        help="verso della punta di B rispetto a quella di A. 'entrambe' raddoppia le partenze "
                             "e toglie da Phi il premio sulle punte allineate, cosi' i due versi sono alla pari")
    parser.add_argument("--inclinazioni", default="0",
                        help="inclinazioni (gradi) della punta di B rispetto all'asse di A, es. 0,30,60. "
                             "0 = solo punte parallele (come prima)")
    parser.add_argument("--azimut-step", type=float, default=90.0,
                        help="passo (gradi) delle direzioni in cui inclinare la punta")
    parser.add_argument("--penetrazione", type=float, default=None,
                        help="compenetrazione massima accettata (frazione dei punti); default quella "
                             "dell'ambiente, 0.002. Per mesh chiuse come Fractura: 0.01")
    parser.add_argument("--peso-esterno", type=float, default=0.0,
                        help="penalita' su Phi per il contatto tra superfici esterne (fuori maschera): "
                             "Phi -= peso * frazione di punti esterni a contatto. 0 = disattivata")
    parser.add_argument("--processed", default="examples/data1/processed",
                        help="cartella dei .pt (e di occupancy/) della coppia di ossa")
    parser.add_argument("--out", default="results/runs", help="cartella dei risultati: le pose vanno in <out>/start_search/")
    args = parser.parse_args()

    offsets = [float(o) for o in args.offsets.split(",")]
    rolls = list(np.arange(0.0, 360.0, args.roll_step))
    senses = {"entrambe": [1.0, -1.0], "allineate": [1.0], "opposte": [-1.0]}[args.punte]
    w_tip = 0.0 if args.punte == "entrambe" else None
    # direzioni della punta: parallela (inclinazione 0) e inclinate di ogni
    # angolo di --inclinazioni, in 360/--azimut-step direzioni attorno all'asse
    tilts = [float(x) for x in args.inclinazioni.split(",")]
    dirs = [(t, a) for t in tilts for a in ([0.0] if t == 0 else np.arange(0.0, 360.0, args.azimut_step))]
    tasks = [(float(r), o, args.iters, args.seed * 100003 + k, s, t, float(az)) for k, (s, (t, az), r, o) in
             enumerate((s, d, r, o) for s in senses for d in dirs for r in rolls for o in offsets)]
    print(f"{len(tasks)} partenze ({len(senses)} versi delle punte x {len(dirs)} inclinazioni x {len(rolls)} roll "
          f"x {len(offsets)} scorrimenti), "
          f"{args.iters} iterazioni ciascuna, {args.workers} processi", flush=True)

    results = []
    out_dir = os.path.join(args.out, "start_search")
    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(args.processed, w_tip, args.penetrazione, args.peso_esterno)) as pool:
        for res in pool.imap_unordered(_climb, tasks):
            results.append(res)
            m = res["metrics"]
            print(f"  [{len(results):3d}/{len(tasks)}] punte {'allineate' if res['sense'] > 0 else 'opposte  '} "
                  f"incl {res['tilt']:2.0f}/{res['azim']:3.0f} roll {res['roll']:5.1f} scorrimento {res['offset']:+5.1f} mm: "
                  f"Phi={res['phi']:6.2f} chamfer={m['chamfer']:5.2f} surface={m['surface']:.3f} "
                  f"normal={m['normal']:.2f} ({res['seconds']:.0f}s)", flush=True)

    results.sort(key=lambda r: r["phi"], reverse=True)
    _init_worker(args.processed, w_tip, args.penetrazione, args.peso_esterno)
    os.makedirs(out_dir, exist_ok=True)
    file_a, file_b = _env.dataset.pairs[0]
    np.savez(os.path.join(out_dir, "start_poses.npz"),
             R=np.stack([r["R"] for r in results]), t=np.stack([r["t"] for r in results]),
             phi=np.array([r["phi"] for r in results]), roll=np.array([r["roll"] for r in results]),
             offset=np.array([r["offset"] for r in results]),
             sense=np.array([r["sense"] for r in results]),
             tilt=np.array([r["tilt"] for r in results]), azim=np.array([r["azim"] for r in results]),
             # ossa di questo esempio, per view_pose.py (posa di B = frag_b_rest @ R.T + t)
             frag_a=_env.frag_a.numpy(), frag_b_rest=_env.frag_b_rest.numpy(),
             normals_a=_env.frag_a_normals.numpy(),
             mask_a=_env.fracture_mask_a.numpy(), mask_b=_env.fracture_mask_b.numpy(),
             pair=np.array([file_a, file_b]))

    from infer_rl import save_ply
    print("\nClassifica per Phi (nessuna GT nella scelta):")
    for rank, r in enumerate(results[:5]):
        m = r["metrics"]
        path = os.path.join(out_dir, f"start_rank{rank}.ply")
        save_ply(path, _env.frag_a.numpy(), r["pose"])
        print(f"  rank {rank}: Phi={r['phi']:6.2f} chamfer={m['chamfer']:5.2f} surface={m['surface']:.3f} "
              f"normal={m['normal']:.2f} (punte {'allineate' if r['sense'] > 0 else 'opposte'}, incl {r['tilt']:g}/{r['azim']:g}, roll {r['roll']:g}, "
              f"scorrimento {r['offset']:+g}) -> {path}")
    print(f"\nPose salvate in {os.path.join(out_dir, 'start_poses.npz')} ({file_a} + {file_b}). "
          f"Visualizzazione: python view_pose.py --runs {args.out}")

    if args.confronta_gt:
        from infer_rl import gt_distance
        print("\n[verifica a posteriori] distanza dalla GT:")
        for rank, r in enumerate(results[:5]):
            err, angle = gt_distance(_env, _env.dataset, r["pose"])
            print(f"  rank {rank}: {err:5.1f} mm, {angle:5.1f} gradi")


if __name__ == "__main__":
    main()
