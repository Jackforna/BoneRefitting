"""
Dal dataset Fractura (GARF, file .hdf5) alle coppie .obj usate da questo
progetto, e verifica a posteriori del risultato.

Nel file .hdf5 ogni esempio e' un gruppo <nome>/pieces/<i>/{vertices,faces}
con i pezzi GIA' nella posizione incastrata. Per un test alla cieca:
  - A (pezzo fisso) viene esportato com'e';
  - B viene ruotato a caso attorno al proprio baricentro e spostato di poco
    (senza allontanarlo troppo, perche' preprocess.py normalizza le due ossa
    insieme e la scala in mm dipende dal loro ingombro comune);
  - la trasformazione applicata a B viene salvata a parte (data/gt/<tag>.json)
    e serve SOLO per --valuta, dopo search_start.py.

Il campo shared_faces del file (le facce in comune tra i pezzi) deriva dalla
soluzione: qui e' usato solo da --lista per dire quali pezzi si toccano, non
arriva mai a preprocess.py o alla ricerca.

Uso:
    python fractura_export.py --h5 bone.hdf5 --lista
    python fractura_export.py --h5 bone.hdf5 --nome <nome dell'esempio> --tag osso1
    python fractura_export.py --h5 bone.hdf5 --nome <nome> --pezzi 0 2 --tag osso2
    # ... preprocess.py, occupancy, search_start.py ...
    python fractura_export.py --valuta --gt data/gt/osso1.json --runs runs2 --processed data2/processed
"""
import argparse
import json
import os

import numpy as np

ENV_SCALE = 70.0   # come env.py


def esempi(h5):
    """Tutti i gruppi che contengono 'pieces' (i singoli esempi), con il numero di pezzi."""
    trovati = []

    def visita(nome, oggetto):
        if hasattr(oggetto, "keys") and "pieces" in oggetto:
            trovati.append((nome, len(oggetto["pieces"])))

    h5.visititems(visita)
    return trovati


def pezzi_che_si_toccano(gruppo):
    """Coppie (i, j) di pezzi con facce in comune (solo per scegliere la coppia)."""
    coppie = set()
    for i in gruppo["pieces"]:
        p = gruppo["pieces"][i]
        if "shared_faces" in p:
            for j in np.unique(p["shared_faces"][()]):
                if j >= 0:
                    coppie.add(tuple(sorted((int(i), int(j)))))
    return sorted(coppie)


def scrivi_obj(path, vertici, facce):
    with open(path, "w") as f:
        for v in vertici:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in facce:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


def rotazione_casuale(rng):
    """Rotazione uniforme (quaternione normalizzato da una gaussiana)."""
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def esporta(args):
    import h5py
    with h5py.File(args.h5, "r") as h5:
        if args.lista:
            for nome, n in esempi(h5):
                riga = f"{n:3d} pezzi  {nome}"
                if n > 2:
                    riga += f"   (si toccano: {pezzi_che_si_toccano(h5[nome])})"
                print(riga)
            return
        gruppo = h5[args.nome]
        i, j = args.pezzi
        va, fa = gruppo["pieces"][str(i)]["vertices"][()], gruppo["pieces"][str(i)]["faces"][()]
        vb, fb = gruppo["pieces"][str(j)]["vertices"][()], gruppo["pieces"][str(j)]["faces"][()]
        if (min(i, j), max(i, j)) not in pezzi_che_si_toccano(gruppo) and len(gruppo["pieces"]) > 2:
            print(f"⚠️ I pezzi {i} e {j} non risultano a contatto: scegline due da --lista")

    rng = np.random.default_rng(args.seed)
    R = rotazione_casuale(rng)
    c = vb.mean(axis=0)
    diag = np.linalg.norm(vb.max(axis=0) - vb.min(axis=0))
    direzione = rng.normal(size=3)
    t = direzione / np.linalg.norm(direzione) * 0.3 * diag
    vb_mossa = (vb - c) @ R.T + c + t      # B nella posa di partenza (sconosciuta al programma)

    os.makedirs(args.raw, exist_ok=True)
    os.makedirs(args.gt_dir, exist_ok=True)
    nome_a, nome_b = f"{args.tag}_A.obj", f"{args.tag}_B.obj"   # ordine alfabetico: A fisso, B mobile
    scrivi_obj(os.path.join(args.raw, nome_a), va, fa)
    scrivi_obj(os.path.join(args.raw, nome_b), vb_mossa, fb)
    gt_path = os.path.join(args.gt_dir, f"{args.tag}.json")
    with open(gt_path, "w") as f:
        json.dump({"h5": args.h5, "nome": args.nome, "pezzi": [i, j], "seed": args.seed,
                   "R": R.tolist(), "c": c.tolist(), "t": t.tolist(),
                   "nota": "B esportato = (B_vero - c) @ R.T + c + t"}, f, indent=2)
    print(f"Esportati {os.path.join(args.raw, nome_a)} e {os.path.join(args.raw, nome_b)}")
    print(f"Soluzione salvata a parte in {gt_path} (non usarla prima di aver giudicato rank0)")
    print(f"Prossimo passo: python srcRL/preprocess.py {nome_a} {nome_b} --out <dataN>/processed")


def valuta(args):
    """Distanza tra le pose di search_start.py e la soluzione vera (solo a posteriori)."""
    with open(args.gt) as f:
        gt = json.load(f)
    R, c, t = np.array(gt["R"]), np.array(gt["c"]), np.array(gt["t"])
    with open(os.path.join(args.processed, "normalization.json")) as f:
        norm = json.load(f)
    centro, raggio = np.array(norm["centroid"]), norm["radius"]
    data = np.load(os.path.join(args.runs, "start_search", "start_poses.npz"))
    if "frag_b_rest" in data.files:
        b_rest = data["frag_b_rest"]
    else:
        # file salvato senza le ossa: i punti di B si rileggono dai .pt dello stesso esempio
        from srcRL.dataset import BoneFragmentsDataset
        dataset = BoneFragmentsDataset(processed_dir=args.processed)
        _, b_full, _ = dataset[0]
        b_full = b_full.x if hasattr(b_full, "x") else b_full
        b_rest = b_full[:, :3].numpy() * ENV_SCALE
        print(f"(punti di B letti da {args.processed}/{dataset.pairs[0][1]})")

    # punti di B nel frame del programma -> coordinate .obj -> posa vera -> frame del programma
    b_obj = b_rest / ENV_SCALE * raggio + centro
    b_vero = (b_obj - c - t) @ R + c
    b_vero_env = (b_vero - centro) / raggio * ENV_SCALE

    print(f"Esempio: {gt['nome']} (pezzi {gt['pezzi']})  |  1 unita' del programma = {raggio / ENV_SCALE:.3g} unita' .obj")
    for rank in range(min(args.quanti, len(data["phi"]))):
        posa = b_rest @ data["R"][rank].T + data["t"][rank]
        errore = np.linalg.norm(posa - b_vero_env, axis=1).mean()
        pc, qc = b_vero_env - b_vero_env.mean(0), posa - posa.mean(0)
        u, _, vt = np.linalg.svd(pc.T @ qc)
        d = np.sign(np.linalg.det(vt.T @ u.T))
        Rd = vt.T @ np.diag([1, 1, d]) @ u.T
        angolo = np.degrees(np.arccos(np.clip((np.trace(Rd) - 1) / 2, -1, 1)))
        print(f"  rank {rank}: Phi={data['phi'][rank]:6.2f} -> dalla soluzione {errore:6.2f} "
              f"(unita' del programma, ~mm), {angolo:5.1f} gradi")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", help="file .hdf5 di Fractura")
    parser.add_argument("--lista", action="store_true", help="elenca gli esempi e il numero di pezzi")
    parser.add_argument("--nome", help="esempio da esportare (come scritto da --lista)")
    parser.add_argument("--pezzi", type=int, nargs=2, default=[0, 1], help="indici dei due pezzi (A fisso, B mobile)")
    parser.add_argument("--tag", default="fractura", help="prefisso dei file esportati")
    parser.add_argument("--raw", default="data/raw", help="dove salvare i .obj")
    parser.add_argument("--gt-dir", default="data/gt", help="dove salvare la soluzione (lontano dai dati)")
    parser.add_argument("--seed", type=int, default=0, help="per la posa casuale di B")
    parser.add_argument("--valuta", action="store_true", help="confronta le pose trovate con la soluzione")
    parser.add_argument("--gt", help="con --valuta: file .json della soluzione")
    parser.add_argument("--runs", default="runs", help="con --valuta: cartella dei risultati")
    parser.add_argument("--processed", default="data/processed", help="con --valuta: cartella dei dati dell'esempio")
    parser.add_argument("--quanti", type=int, default=5, help="con --valuta: quante pose della classifica")
    args = parser.parse_args()

    if args.valuta:
        valuta(args)
    elif args.h5 and (args.lista or args.nome):
        esporta(args)
    else:
        parser.error("usa --h5 FILE --lista, --h5 FILE --nome NOME, oppure --valuta --gt FILE")


if __name__ == "__main__":
    main()
