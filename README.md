# 3D Bone Refitting

Ricomposizione di due frammenti ossei (A fisso, B mobile) con Reinforcement
Learning (PPO, stable-baselines3) e una GNN sull'interfaccia di frattura.
La ground truth (`srcRL/ground_truth.py`) NON e' usata dal modello: serve
solo come verifica a posteriori (`--confronta-gt`).

## Struttura

| File | Cosa fa |
|---|---|
| `env.py` | Ambiente Gymnasium: movimenti di B, metriche, reward (potential-based shaping su Phi), compenetrazione reale, partenze |
| `trainRL.py` | Addestramento PPO (checkpoint, ripresa, log, opzioni `--face`, `--roll`, `--all-rolls`) |
| `compare_runs.py` | Confronta i run in `runs/` e li ordina per Phi |
| `infer_rl.py` | Fa partire un modello addestrato da piu' roll e salva la posa con il Phi piu' alto |
| `preview_starts.py` | Salva le 6 partenze sul cubo attorno ad A (.ply + immagine) |
| `srcRL/InterfaceEncoder.py` | GNN del grafo d'interfaccia A-B, addestrata dal PPO |
| `srcRL/occupancy.py` | Griglie dentro/fuori delle mesh, per misurare la compenetrazione reale |
| `srcRL/roll_sampler.py` | Scelta adattiva del roll di partenza in base a Phi |
| `srcRL/callbacks.py` | Render e record del miglior incastro durante l'addestramento |
| `srcRL/dataset.py` | Caricamento dei frammenti preprocessati |
| `srcRL/preprocess.py` | Preprocessing delle mesh (.obj -> .pt con maschere di frattura) |

Dati necessari:
- `data/processed/1776a.pt`, `data/processed/2480.pt` (da `srcRL/preprocess.py`)
- `data/processed/occupancy/1776a.npz`, `2480.npz` (da `python -m srcRL.occupancy`, ~10 minuti)

## Potenziale Phi

Qualita' della posa, usata sia nel reward (Phi dopo - Phi prima) sia per
scegliere tra piu' tentativi:

    Phi = 10*tip + 10*normal + 100*surface - min(chamfer, 60)

Le pose con compenetrazione sopra lo 0.2% non sono mai considerate valide.

## Procedura: ricerca faccia + roll

L'agente non sa correggere grandi rotazioni di B attorno al proprio asse
lungo (roll) una volta che le ossa si toccano. Si addestra quindi un
modello per ogni faccia del cubo attorno ad A (davanti, dietro, destra,
sinistra, sopra, sotto), con B a punte allineate e 8 roll di partenza, e
si sceglie la coppia faccia + roll con il Phi migliore.

1. Controllare le partenze (facoltativo):

       python preview_starts.py

2. Un run per faccia (3-4 in parallelo, uno per terminale):

       python trainRL.py --face sopra    --all-rolls --steps 150000 --no-viewer
       python trainRL.py --face sotto    --all-rolls --steps 150000 --no-viewer
       python trainRL.py --face destra   --all-rolls --steps 150000 --no-viewer
       python trainRL.py --face sinistra --all-rolls --steps 150000 --no-viewer
       python trainRL.py --face davanti  --all-rolls --steps 150000 --no-viewer
       python trainRL.py --face dietro   --all-rolls --steps 150000 --no-viewer

   Risultati in `runs/face_<faccia>_rolls/`.

3. Classifica per Phi:

       python compare_runs.py --episodi 3 --tieni 2

4. Continuare solo la coppia vincente, con partenza fissa:

       python trainRL.py --face <faccia> --roll <roll> --steps 400000 --no-viewer

5. Risultato finale:

       python infer_rl.py --model runs/face_<faccia>_roll_<roll>/ppo_archaeological_aligner.zip --vecnorm runs/face_<faccia>_roll_<roll>/ppo_archaeological_aligner_vecnormalize.pkl --rolls <roll> --episodi-per-roll 5

   La posa viene salvata in `output_3d/infer_best.ply`. Aggiungendo
   `--confronta-gt` viene stampata anche la distanza dalla GT.

## Opzioni di trainRL.py

| Opzione | Effetto |
|---|---|
| `--face F` | B parte dalla faccia F del cubo, punte allineate (roll 0 se non indicato) |
| `--roll R` | roll di partenza fisso R (gradi) |
| `--all-rolls` | con `--face`: 8 roll con scelta adattiva per Phi |
| `--steps N` | numero totale di step |
| `--resume N` | riprende dal checkpoint di N step dello stesso run |
| `--no-viewer` | non apre la finestra Open3D a fine addestramento |

Senza opzioni valgono le costanti in cima a `trainRL.py` e i risultati
vanno nella cartella principale.

Log utili (tabella di SB3): `episode/final_chamfer`, `episode/best_phi`,
`episode/success_rate`, `roll/best_roll`, `roll/best_roll_prob`.
I checkpoint (ogni 100k step) contengono modello, statistiche di
normalizzazione e record del miglior incastro.
