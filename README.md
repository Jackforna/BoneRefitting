# BoneRefitting

Progetto per ricomporre due frammenti di osso scansionati in 3D. Il frammento A resta fermo e il frammento B viene spostato e ruotato finché non trova la posizione in cui si incastra con A. Il programma non usa la soluzione (ground truth), che serve solo per controllare i risultati alla fine.

Il metodo finale è una ricerca con molte partenze: B viene messo vicino ad A in tante posizioni iniziali diverse (rotazione attorno al proprio asse, inclinazione, verso, spostamento lungo A) e da ognuna fa piccoli movimenti casuali, tenendo solo quelli che migliorano il punteggio Φ senza far entrare un osso dentro l'altro. Alla fine vince la posa con Φ più alto.

All'inizio avevo provato con il Reinforcement Learning (PPO), quella parte è nella cartella `RL_attempt/`.

## Installazione

Ho usato Python 3.10/3.11.

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Tutti i comandi vanno lanciati dalla cartella principale del progetto.

## Cartelle e file

- `search_start.py`: la ricerca della posa (metodo finale)
- `env.py`: calcolo delle metriche, di Φ e della compenetrazione tra le ossa
- `view_pose.py`: per vedere in 3D le pose trovate
- `infer_rl.py`: per provare un modello PPO già addestrato
- `src/preprocess.py`: dai file `.obj` ai punti con normali e maschera di frattura (`.pt`)
- `src/occupancy.py`: griglie dentro/fuori delle mesh, servono per capire se le ossa si compenetrano
- `src/gt_check.py`: crea il file della ground truth e lo usa per il confronto finale
- `src/dataset.py`: carica i file `.pt`
- `RL_attempt/`: tutta la parte di Reinforcement Learning
- `examples/dataN/`: dati della coppia N (`raw/` con le mesh, `processed/` con i file preprocessati)
- `results/runsN/`: pose trovate per la coppia N

Le coppie sono:
1. 1776a + 2480
2. 2347 + 3045
3. D1550 + D7428
4. 1960 + N1248

Le mesh `3045.obj`, `1960.obj` e `N1248.obj` pesano più di 100 MB e GitHub non le accetta, quindi non sono nel repository. Per rifare il preprocessing delle coppie 2 e 4 bisogna copiarle a mano in `examples/data2/raw/` e `examples/data4/raw/`. I file già preprocessati ci sono, quindi la ricerca si può lanciare lo stesso.

## Come si usa

Esempio con la coppia 1.

1. Preprocessing (a fine esecuzione si apre una finestra con le maschere, chiuderla per continuare):

```
python -m src.preprocess 1776a.obj 2480.obj --data 1
```

2. Griglie di occupazione. Vanno rifatte ogni volta che si rifà il preprocessing:

```
python -m src.occupancy --data 1
```

3. Ricerca della posa:

```
python search_start.py --processed examples/data1/processed --out results/runs1 --inclinazioni 0,30,60 --roll-step 30 --workers 4
```

Con questi parametri sono 648 partenze. Sulla coppia 1 ci mette circa un'ora con 4 core, sulle coppie con mesh più grandi anche di più. Con meno partenze (`--roll-step 15 --inclinazioni 0`, 72 partenze) ci mette 15-20 minuti ma non trova le ossa inclinate tra loro.

4. Visualizzazione:

```
python view_pose.py --runs 1
python view_pose.py --runs 1 --rank 1
```

`--rank 0` è la posa migliore, `--rank 1` la seconda e così via.

Il campionamento dei punti nel preprocessing ha il seed fisso, quindi rifacendo tutto con gli stessi file si ottengono gli stessi risultati. Se si rifà il preprocessing però bisogna rifare anche la ricerca, perché le pose salvate fanno riferimento ai punti di prima.

## Confronto con la ground truth

Le ground truth le ho fatte a mano con CloudCompare, quindi sono approssimate. Il confronto si fa solo alla fine e non influisce sulla scelta della posa.

```
python search_start.py --processed examples/data1/processed --out results/runs1 --confronta-gt
```


## Parte di Reinforcement Learning

```
python RL_attempt/preview_starts.py
python RL_attempt/trainRL.py --face sopra --all-rolls --steps 150000 --no-viewer
python RL_attempt/compare_runs.py
python infer_rl.py --processed examples/data1/processed
```
```
I modelli già addestrati sono in `RL_attempt/checkpoints/`. Questa parte l'ho provata solo sulla coppia 1: l'agente trova l'incastro solo se parte già molto vicino alla soluzione, per questo sono passato alla ricerca con più partenze.
```