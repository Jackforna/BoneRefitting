"""
Estrae dal file .hdf5 di Fractura una versione alleggerita delle mesh di
tutti i pezzi, per addestrare il classificatore della superficie di frattura.

Per ogni esempio (gruppo con 'pieces') salva i vertici e le facce di ogni
pezzo, semplificati a --triangoli triangoli, nella posizione incastrata del
file: serve per ricavare le etichette di addestramento (punti a contatto con
un altro pezzo). I pezzetti staccati (meno dell'1% dei triangoli, es.
marcatori) vengono tolti. Il campo shared_faces del file NON viene usato.

Uso:
    python export_training_meshes.py --h5 fractura/bone_real.hdf5 --out data/training_meshes.npz
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


def alleggerisci(v, f, n_triangoli):
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
    clusters, counts, _ = mesh.cluster_connected_triangles()
    clusters, counts = np.asarray(clusters), np.asarray(counts)
    piccoli = counts < 0.01 * counts.sum()
    if piccoli.any():
        mesh.remove_triangles_by_mask(piccoli[clusters])
        mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > n_triangoli:
        mesh = mesh.simplify_quadric_decimation(n_triangoli)
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.triangles, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--out", default=os.path.join("data", "training_meshes.npz"))
    parser.add_argument("--triangoli", type=int, default=30000, help="triangoli per pezzo dopo la semplificazione")
    args = parser.parse_args()

    import h5py
    dati, nomi = {}, []
    with h5py.File(args.h5, "r") as h5:
        lista = esempi(h5)
        print(f"{len(lista)} esempi nel file")
        for e, nome in enumerate(lista):
            pezzi = h5[nome]["pieces"]
            for i in sorted(pezzi, key=int):
                v, f = alleggerisci(pezzi[i]["vertices"][()], pezzi[i]["faces"][()], args.triangoli)
                dati[f"e{e}_p{i}_v"] = v
                dati[f"e{e}_p{i}_f"] = f
            nomi.append(nome)
            print(f"  [{e + 1}/{len(lista)}] {nome}: {len(pezzi)} pezzi")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, nomi=np.array(nomi), **dati)
    print(f"Salvato {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
