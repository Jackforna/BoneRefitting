"""
Estrae dal file .hdf5 di Fractura una versione alleggerita delle mesh di
tutti i pezzi, per addestrare il classificatore della superficie di frattura.

Per ogni esempio (gruppo con 'pieces') salva i vertici e le facce di ogni
pezzo, semplificati a --triangoli triangoli, nella posizione incastrata del
file: serve per ricavare le etichette di addestramento (punti a contatto con
un altro pezzo). I pezzetti staccati (meno dell'1% dei triangoli, es.
marcatori) vengono tolti. Il campo shared_faces del file NON viene usato.

Uso:
    python export_training_meshes.py --h5 fractura/bone_real.hdf5 --out fractura/data/training_meshes.npz
"""
import argparse
import os

import numpy as np
import open3d as o3d


def esempi(h5):
    trovati = []

    def visita(nome, oggetto):
        if hasattr(oggetto, "keys") and "pieces" in oggetto:
            trovati.append(nome)

    h5.visititems(visita)
    return trovati


def componenti(f, n_vertici):
    """Etichetta di componente connessa per ogni triangolo (triangoli uniti se condividono un vertice)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    righe = np.repeat(np.arange(len(f)), 3)
    g = coo_matrix((np.ones(len(righe)), (righe, f.ravel())), shape=(len(f), n_vertici)).tocsr()
    adiacenza = g @ g.T
    return connected_components(adiacenza, directed=False)[1]


def salda(v, f, tolleranza=1e-6):
    """Unisce i vertici coincidenti (a meno di tolleranza * dimensione del pezzo) e toglie i triangoli degeneri."""
    passo = tolleranza * np.ptp(v, axis=0).max()
    chiavi = np.round(v / passo).astype(np.int64)
    _, primo, inverso = np.unique(chiavi, axis=0, return_index=True, return_inverse=True)
    f = inverso.ravel()[f]
    buoni = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    return v[primo], f[buoni]


def alleggerisci(v, f, n_triangoli):
    v = np.ascontiguousarray(v, dtype=np.float64)
    f = np.ascontiguousarray(f, dtype=np.int64)
    # alcune mesh hanno vertici separati per ogni triangolo: prima si saldano
    v, f = salda(v, f)
    # pezzetti staccati (marcatori): via i componenti con meno dell'1% dei triangoli
    comp = componenti(f, len(v))
    conta = np.bincount(comp)
    f = f[conta[comp] >= 0.01 * len(f)]
    usati = np.unique(f)
    nuovo = -np.ones(len(v), dtype=np.int64); nuovo[usati] = np.arange(len(usati))
    v, f = v[usati], nuovo[f]
    if len(f) <= n_triangoli:
        return v.astype(np.float32), f.astype(np.int32)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f.astype(np.int32)))
    ridotta = mesh.simplify_quadric_decimation(target_number_of_triangles=n_triangoli)
    if len(ridotta.triangles) == 0:
        # riserva: raggruppamento dei vertici su una griglia
        lato = np.ptp(v, axis=0).max() / 150
        ridotta = mesh.simplify_vertex_clustering(voxel_size=lato)
    return np.asarray(ridotta.vertices, dtype=np.float32), np.asarray(ridotta.triangles, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--out", default=os.path.join("fractura/data", "training_meshes.npz"))
    parser.add_argument("--triangoli", type=int, default=30000, help="triangoli per pezzo dopo la semplificazione")
    args = parser.parse_args()

    import h5py
    dati, nomi = {}, []
    with h5py.File(args.h5, "r") as h5:
        lista = esempi(h5)
        print(f"{len(lista)} esempi nel file")
        for e, nome in enumerate(lista):
            pezzi = h5[nome]["pieces"]
            righe = []
            for i in sorted(pezzi, key=int):
                v0, f0 = pezzi[i]["vertices"][()], pezzi[i]["faces"][()]
                v, f = alleggerisci(v0, f0, args.triangoli)
                if len(f) == 0:
                    raise SystemExit(f"{nome} pezzo {i}: mesh vuota dopo la semplificazione "
                                     f"(in ingresso {v0.shape} {v0.dtype}, {f0.shape} {f0.dtype})")
                dati[f"e{e}_p{i}_v"] = v
                dati[f"e{e}_p{i}_f"] = f
                righe.append(f"p{i}: {len(f0)} -> {len(f)} triangoli")
            nomi.append(nome)
            print(f"  [{e + 1}/{len(lista)}] {nome}: " + ", ".join(righe))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, nomi=np.array(nomi), **dati)
    print(f"Salvato {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
