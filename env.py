import os
import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.cluster import KMeans

from RL_attempt.InterfaceEncoder import IFACE_NUM_FEATURES
from src.occupancy import OccupancyGrid
from src.roll_sampler import RollSampler


class ArchaeologicalIncastroEnv(gym.Env):
    """
    Ambiente RL per l'incastro di due frammenti ossei (A fisso, B mobile).

    obs_mode:
      - "interface" (default): osservazione Dict con
          "geom":  metriche geometriche (vedi _get_obs)
          "iface": nodi del grafo congiunto dell'interfaccia di frattura
                   A-B nella posa CORRENTE, ricalcolati ad ogni step.
        Da usare con MultiInputPolicy + InterfaceFeaturesExtractor
        (srcRL/InterfaceEncoder.py): la GNN viene addestrata dal PPO.
      - "geom": solo il vettore delle metriche geometriche, senza alcun
        encoder (da usare con MlpPolicy). Utile come confronto.
      - "legacy": vettore piatto [geom, embedding del GraphEncoder].
        L'embedding e' calcolato una sola volta per episodio sulle ossa
        nella posa originale del dataset, quindi con una sola coppia e'
        costante: resta solo per confronto con i modelli precedenti.
    """
    metadata = {"render_modes": ["human"]}

    # Scale (mm) usate per normalizzare le feature dei nodi di interfaccia
    IFACE_POS_SCALE = 30.0
    IFACE_DIST_SCALE = 5.0

    def __init__(self, dataset, encoder=None, max_steps=120, max_allowed_chamfer=100.0,
                 obs_mode="interface", n_iface_points=128, device=None,
                 occupancy_dir="data3/processed/occupancy", shaping_gamma=1.0,
                 start_mode="tip_aligned", start_rolls=None, start_face=None,
                 start_poses=None, start_rank=0):
        super(ArchaeologicalIncastroEnv, self).__init__()

        if obs_mode not in ("interface", "legacy", "geom"):
            raise ValueError(f"obs_mode non valido: {obs_mode}")
        if obs_mode == "legacy" and encoder is None:
            raise ValueError("obs_mode='legacy' richiede un encoder")

        self.dataset = dataset
        self.encoder = encoder
        if self.encoder is not None:
            self.encoder.eval()
        self.obs_mode = obs_mode
        self.n_iface_points = n_iface_points
        self.occupancy_dir = occupancy_dir
        self.shaping_gamma = shaping_gamma

        # Posa di partenza di B a ogni reset (nessuna delle due usa la GT):
        #  - "random": rotazione uniforme su SO(3) attorno al baricentro.
        #  - "tip_aligned": riproduce la struttura della vecchia partenza
        #    T_GT ricavandola solo dalla geometria delle due ossa: punte
        #    allineate (tip_alignment ~ +1), roll attorno all'asse lungo
        #    CASUALE, piccola inclinazione casuale, B a START_DISTANCE mm
        #    dalla maschera di frattura di A lungo la sua normale media.
        #    Il roll e' l'unico grado di liberta' che l'agente non sa
        #    correggere una volta a contatto: in inferenza si provano piu'
        #    roll (reset(options={"roll_deg": ...})) e si tiene la posa con
        #    il potenziale Phi piu' alto (vedi infer_rl.py).
        #  - "cube": punte allineate come "tip_aligned", ma B posizionato su
        #    una delle 6 facce di un cubo attorno ad A (vedi CUBE_FACES),
        #    fuori dal suo ingombro di CUBE_MARGIN mm. start_face fissa la
        #    faccia (None = casuale a ogni episodio), oppure
        #    reset(options={"face": ...}).
        #  - "searched": la posa trovata da search_start.py (ricerca su Phi,
        #    senza GT), staccata di START_DISTANCE mm lungo la normale media
        #    della maschera di A, come la vecchia partenza T_GT.
        #    start_poses = {"R": (K,3,3), "t": (K,3)} in classifica per Phi,
        #    start_rank sceglie quale (0 = Phi migliore), oppure
        #    reset(options={"rank": ...}).
        if start_mode not in ("random", "tip_aligned", "cube", "searched"):
            raise ValueError(f"start_mode non valido: {start_mode}")
        self.start_mode = start_mode
        self.START_DISTANCE = 15.0      # mm, come la vecchia partenza T_GT
        self.CUBE_MARGIN = 10.0         # mm di distacco tra gli ingombri di A e B (start_mode="cube")
        if start_face is not None and start_face not in self.CUBE_FACES:
            raise ValueError(f"start_face non valida: {start_face} (possibili: {list(self.CUBE_FACES)})")
        self.start_face = start_face
        self.episode_face = None
        if start_mode == "searched" and start_poses is None:
            raise ValueError("start_mode='searched' richiede start_poses (vedi search_start.py)")
        self.start_poses = start_poses
        self.start_rank = start_rank
        self.START_MAX_TILT = 10.0      # gradi di inclinazione casuale dell'asse
        self.START_MAX_AXIAL_SHIFT = 10.0  # mm di scorrimento casuale lungo l'asse di A
        self.start_roll_deg = None
        self.episode_rank = None
        # start_rolls: lista di roll (gradi) fissi per le partenze
        # tip_aligned, scelti in modo adattivo in base al Phi raggiunto
        # (srcRL/roll_sampler.py). None = roll continuo uniforme.
        self.roll_sampler = RollSampler(start_rolls) if start_rolls is not None else None
        self._roll_index = None
        self.episode_best_phi = None
        self._occupancy_cache = {}
        self.occupancy_a = None
        self.occupancy_b = None
        self.frag_b_rest = None
        self.max_steps = max_steps
        self.max_allowed_chamfer = max_allowed_chamfer
        self.current_step = 0
        if device is not None:
            self.device = torch.device(device)
        elif self.encoder is not None:
            self.device = next(self.encoder.parameters()).device
        else:
            self.device = torch.device("cpu")

        self.frag_a = None
        self.frag_b = None
        self.frag_a_normals = None
        self.frag_b_normals = None

        # RECORD GLOBALI (persistono per tutto l'addestramento)
        self.global_best_chamfer = float('inf')
        self.global_best_normal = 0.0
        self.global_best_surface_matching_ratio = 0.0  # gate progressivo su coverage
        self.global_best_penetration_ratio = float('inf')
        self.best_frag_a = None
        self.best_frag_b = None
        self.best_chamfer = float('inf')
        self.best_normal = 0.0
        self.best_surface_matching_ratio = 0.0
        self.best_penetration_ratio = float('inf')

        self.fracture_mask_a = None
        self.fracture_mask_b = None
        self.shape_emb = None

        self.current_rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.max_trans_step = 0.5
        self.max_rot_step = 0.05

        self.collision_threshold = 0.65

        # Tolleranza sul numero di punti compenetranti prima del
        # rollback completo della posizione. Prima, QUALSIASI punto
        # compenetrante (anche uno solo) annullava tutto il movimento
        # -- questo puo' bloccare sul nascere una transizione che
        # richiede un contatto temporaneo minimo per raggiungere una
        # configurazione complessivamente migliore. Sotto questa
        # soglia, il movimento resta valido e viene penalizzato in
        # modo continuo tramite penalty_penetration, non annullato di
        # netto. Non validato su GT (riguarda la dinamica di
        # esplorazione, non un criterio discriminativo).
        self.COLLISION_TOLERANCE_CONTACTS = 10

        # Compenetrazione REALE, misurata con le griglie di occupazione
        # (srcRL/occupancy.py): frazione dei punti di A e di B che
        # finiscono dentro il materiale dell'altro osso. Il vecchio
        # controllo (distanza < 0.65 mm + segno della normale) non vedeva
        # i punti affondati in profondita', e l'agente otteneva chamfer
        # e surface migliori della GT facendo compenetrare le ossa.
        #
        # Valori misurati: GT calibrata 0.0007 (0.07%); risultati
        # visivamente sbagliati del training precedente 0.03-0.06.
        # - PENETRATION_ROLLBACK_RATIO: sopra questa soglia il movimento
        #   viene annullato (sostituisce COLLISION_TOLERANCE_CONTACTS).
        # - PENETRATION_PENALTY_WEIGHT: penalita' continua sotto la soglia
        #   di rollback (-1 di reward allo 0.5% di punti compenetrati).
        self.PENETRATION_ROLLBACK_RATIO = 0.005
        self.PENETRATION_PENALTY_WEIGHT = 200.0

        # Soglie CALIBRATE empiricamente sul campione con ground truth
        # nota (vedi ground_truth.py), non scelte a intuito:
        #
        # - NORMAL_ALIGNMENT_TARGET: alla posa GT vera, normal_alignment
        #   (locale, sulla maschera statica) vale 0.7349. Su 20 pose
        #   sbagliate casuali, il massimo osservato e' 0.5563. 0.65 sta
        #   comodamente in mezzo.
        #
        # - PENETRATION_RATIO_TARGET: alla posa GT vera, penetration
        #   vale 0.0112. Su 20 pose sbagliate casuali, il minimo
        #   osservato e' 0.0298. 0.02 sta comodamente in mezzo.
        #
        # current_chamfer (sulla maschera statica) NON e' usato come
        # criterio di successo/gate: resta >20mm anche alla posa GT
        # perfetta -- irraggiungibile per costruzione con questa
        # maschera. Resta solo come metrica informativa nella reward.
        self.NORMAL_ALIGNMENT_TARGET = 0.65
        # Ricalibrato sulla compenetrazione reale (griglie di
        # occupazione): GT = 0.0007, risultati sbagliati 0.03-0.06.
        self.PENETRATION_RATIO_TARGET = 0.002
        self.CONTACT_MIN_DIST_TARGET = self.collision_threshold

        # RICALIBRATO dopo il fix del bug che ha attivato la formula
        # blend (60% media + 40% minimo tra settori): valore vero alla
        # GT = 0.1180 (formula attuale, non piu' 0.0698 della formula
        # vecchia a minimo puro). Su 20 pose sbagliate: media 0.0554,
        # solo 1/20 sopra il valore GT (0.1761, outlier). 0.08 sta in
        # mezzo, con margine su entrambi i lati.
        self.SURFACE_MATCHING_TARGET = 0.08

        # Validato su GT: overlap assiale (asse principale di A) alla
        # posa corretta = 0.9296, minimo su 20 pose sbagliate = 0.9730
        # -- margine sottile ma consistente (0/20 sotto il valore GT).
        # Soglia poco sopra il valore GT, per non penalizzare la posa
        # corretta stessa.
        self.AXIAL_OVERLAP_SOFT_THRESHOLD = 0.95

        # 🆕 Validato su GT: dot(direzione punta A, direzione punta B)
        # -- vettore baricentro->punto piu' lontano, un segnale GLOBALE
        # di orientamento (non locale come normal_alignment, che guarda
        # solo il punto di contatto e resta cieco a ossa orientate in
        # direzioni opposte). Alla posa GT vale 0.9965 (punte quasi
        # perfettamente allineate). Su 20 pose sbagliate: range
        # [-0.991, 0.981], media -0.048, 0/20 raggiungono il valore GT.
        # 0.7 sta ben sopra dove finirebbero orientazioni opposte o
        # perpendicolari (vicino a -1 o 0), con margine per il rumore
        # di allenamento.
        self.TIP_ALIGNMENT_TARGET = 0.7

        # Soglia (non validata su GT, euristica basata sui valori
        # osservati) sotto la quale consideriamo il chamfer "gia'
        # abbastanza buono" e spostiamo il peso della reward sulla
        # coverage.
        self.CHAMFER_GOOD_ENOUGH = 13.0  # 6 se CHAMFER_KEEP_FRACTION = 0.2

        # ------------------------------------------------------------
        # REWARD con potential-based shaping (Ng et al., 1999).
        #
        # Prima tip_alignment*10 e surface*w_coverage venivano dati AD
        # OGNI STEP: restare fermi con le punte allineate valeva ~+10 per
        # step (+5000 a episodio), mentre il successo valeva +100 una
        # volta sola e chiudeva l'episodio. L'agente imparava quindi a
        # orientare B e a "parcheggiarlo" lontano, senza mai incastrarlo.
        #
        # Ora la qualita' della posa e' un potenziale Phi(posa) e il
        # reward di shaping e' Phi(dopo) - Phi(prima): si guadagna solo
        # MIGLIORANDO la posa, si perde peggiorandola, e restare fermi
        # rende esattamente 0. Il totale su un episodio dipende solo da
        # posa iniziale e finale.
        #
        # shaping_gamma = 1 (e non il gamma del PPO): con gamma < 1 una
        # posa ferma con Phi negativo guadagnava -(1-gamma)*Phi > 0 per
        # step, cioe' parcheggiare LONTANO rendeva piu' che parcheggiare
        # vicino (verificato: +50 a 40 mm contro +10 a 20 mm).
        #
        # Phi = W_TIP*tip + W_NORMAL*normal + W_COVERAGE*surface
        #       - W_CHAMFER*min(chamfer, CHAMFER_POTENTIAL_CAP)
        # Valori misurati: GT ~ +21, posa orientata ma a 20 mm ~ -23,
        # a 40 mm ~ -52.
        self.W_TIP = 10.0
        self.W_NORMAL = 0.0 #10.0 se si vuole mettere
        self.W_COVERAGE = 100.0
        self.W_CHAMFER = 1.0
        self.CHAMFER_POTENTIAL_CAP = 60.0

        # Costo fisso per step: finire prima conviene.
        self.STEP_PENALTY = 0.05
        self.SUCCESS_REWARD = 100.0

        # Dimensione osservazione geometrica (+1 per tip_alignment)
        base_obs_dim = 16
        self.iface_idx_a = None
        self.iface_idx_b = None

        if self.obs_mode == "interface":
            self.observation_space = spaces.Dict({
                "geom": spaces.Box(low=-np.inf, high=np.inf, shape=(base_obs_dim,), dtype=np.float32),
                "iface": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(2 * self.n_iface_points, IFACE_NUM_FEATURES), dtype=np.float32,
                ),
            })
        elif self.obs_mode == "geom":
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(base_obs_dim,), dtype=np.float32)
        else:
            _, _, (dummy_graph_a, dummy_graph_b) = self.dataset[0]
            with torch.no_grad():
                dummy_emb = self._encode_pair(dummy_graph_a, dummy_graph_b)
            self.emb_dim = dummy_emb.shape[0]
            total_obs_dim = base_obs_dim + self.emb_dim
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32)

        self.prev_chamfer = float('inf')
        self.prev_normal = 0.0
        self.episode_best_chamfer = float('inf')
        self.episode_best_normal = 0.0
        self.episode_best_surface_matching_ratio = 0.0

    @staticmethod
    def _orient_normals_outward(points, normals):
        centroid = points.mean(dim=0, keepdim=True)
        outward_ref = points - centroid
        outward_ref = outward_ref / (torch.norm(outward_ref, dim=1, keepdim=True) + 1e-8)

        dot = torch.sum(normals * outward_ref, dim=1, keepdim=True)
        sign = torch.where(dot < 0, -torch.ones_like(dot), torch.ones_like(dot))

        return normals * sign

    @staticmethod
    def _tip_direction(points):
        """
        Direzione "punta": vettore dal baricentro al punto piu'
        lontano, normalizzato. A differenza dell'asse principale
        (PCA), non ha ambiguita' di segno -- e' definito da un punto
        specifico (il piu' lontano), non da un asse bidirezionale.
        Segnale GLOBALE di orientamento del corpo del frammento,
        indipendente da dove si toccano i punti di frattura.
        """
        centroid = points.mean(dim=0)
        dists = torch.norm(points - centroid, dim=1)
        idx = torch.argmax(dists)
        direction = points[idx] - centroid
        return direction / (torch.norm(direction) + 1e-8)

    @staticmethod
    def _random_rotation_matrix(device, dtype):
        """
        Rotazione casuale UNIFORME su SO(3), via quaternione
        normalizzato campionato da una gaussiana.

        🆕 FIX: la versione precedente campionava 3 angoli di Eulero
        indipendenti e uniformi in [0, 2*pi) -- questo NON produce una
        distribuzione uniforme sulle rotazioni 3D (problema noto di
        parametrizzazione: la densita' non e' uniforme rispetto alla
        misura di Haar su SO(3), certe orientazioni vengono
        sovracampionate). Il quaternione gaussiano normalizzato e' il
        modo standard per ottenere una rotazione genuinamente uniforme.
        """
        q = torch.randn(4, device=device, dtype=dtype)
        q = q / torch.norm(q)
        w, x, y, z = q[0], q[1], q[2], q[3]

        R = torch.stack([
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
        ])
        return R

    # Record del miglior incastro trovato: salvati accanto ai checkpoint e
    # ripristinati quando l'addestramento riprende (vedi trainRL.py).
    RECORD_FIELDS = (
        "global_best_chamfer", "global_best_normal", "global_best_surface_matching_ratio",
        "global_best_penetration_ratio", "best_chamfer", "best_normal",
        "best_surface_matching_ratio", "best_penetration_ratio",
    )

    def _end_episode(self):
        if self.roll_sampler is not None and self._roll_index is not None:
            self.roll_sampler.update(self._roll_index, self.episode_best_phi)

    def _start_info(self):
        info = {}
        if self.episode_face is not None:
            info["start_face"] = self.episode_face
        if self.episode_rank is not None:
            info["start_rank"] = int(self.episode_rank)
        if self.start_roll_deg is not None:
            info["start_roll"] = float(self.start_roll_deg)
        if self.episode_best_phi is not None:
            info["episode_best_phi"] = float(self.episode_best_phi)
        if self.roll_sampler is not None:
            best_roll, best_score = self.roll_sampler.best()
            if best_roll is not None:
                info["roll_sampler_best_roll"] = best_roll
                info["roll_sampler_best_score"] = best_score
                info["roll_sampler_best_prob"] = float(
                    self.roll_sampler.probabilities()[self.roll_sampler.rolls.index(best_roll)]
                )
        return info

    def get_records(self):
        records = {f"env_{k}": float(getattr(self, k)) for k in self.RECORD_FIELDS}
        if self.roll_sampler is not None:
            records.update(self.roll_sampler.state())
        if self.best_frag_a is not None:
            records["env_best_frag_a"] = self.best_frag_a
            records["env_best_frag_b"] = self.best_frag_b
        return records

    def set_records(self, records):
        for k in self.RECORD_FIELDS:
            if f"env_{k}" in records:
                setattr(self, k, float(records[f"env_{k}"]))
        if "env_best_frag_a" in records:
            self.best_frag_a = np.array(records["env_best_frag_a"])
            self.best_frag_b = np.array(records["env_best_frag_b"])
        if self.roll_sampler is not None:
            self.roll_sampler.load_state(records)

    @staticmethod
    def _axis_angle(axis, angle, device, dtype):
        axis = axis / (torch.norm(axis) + 1e-8)
        K = torch.zeros(3, 3, device=device, dtype=dtype)
        K[0, 1], K[0, 2] = -axis[2], axis[1]
        K[1, 0], K[1, 2] = axis[2], -axis[0]
        K[2, 0], K[2, 1] = -axis[1], axis[0]
        I = torch.eye(3, device=device, dtype=dtype)
        return I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    # Facce del cubo attorno ad A, sugli assi principali di A:
    # asse lungo (verso la punta di A = "davanti"), larghezza, spessore.
    CUBE_FACES = {
        "davanti": (0, +1.0), "dietro": (0, -1.0),
        "destra": (1, +1.0), "sinistra": (1, -1.0),
        "sopra": (2, +1.0), "sotto": (2, -1.0),
    }

    def cube_axes(self):
        """Assi del cubo: asse lungo di A orientato verso la sua punta, poi larghezza e spessore (terna destrorsa)."""
        e1 = self.principal_axes_a[0]
        if torch.dot(e1, self.tip_dir_a) < 0:
            e1 = -e1
        e2 = self.principal_axes_a[1]
        e3 = torch.linalg.cross(e1, e2)
        return e1, e2, e3 / (torch.norm(e3) + 1e-8)

    def _place_b_on_cube_face(self, face):
        """
        Sposta B (gia' orientato) sulla faccia del cubo indicata: il
        baricentro di B sulla retta che esce dal baricentro di A lungo la
        direzione della faccia, con CUBE_MARGIN mm tra gli ingombri.
        """
        axis_index, sign = self.CUBE_FACES[face]
        with torch.no_grad():
            direction = self.cube_axes()[axis_index] * sign
            center_a = self.frag_a.mean(dim=0)
            center_b = self.frag_b.mean(dim=0)
            extent_a = torch.max((self.frag_a - center_a) @ direction)
            extent_b = torch.max(-(self.frag_b - center_b) @ direction)
            target = center_a + direction * (extent_a + extent_b + self.CUBE_MARGIN)
            self.frag_b = self.frag_b - center_b + target
            self._start_away_dir = direction
        self.episode_face = face

    def _place_b_tip_aligned(self, roll_deg=None):
        """
        Partenza senza GT: punta di B allineata alla punta di A, roll
        (rotazione attorno alla punta) dato o casuale, piccola inclinazione
        casuale, poi B portata a START_DISTANCE mm dalla maschera di
        frattura di A lungo la normale media della maschera.
        """
        dev, dt = self.device, self.frag_b.dtype
        rng = self.np_random
        with torch.no_grad():
            center_b = self.frag_b.mean(dim=0)
            pts = self.frag_b - center_b
            normals = self.frag_b_normals

            # 1) punta di B -> punta di A
            tip_b = self._tip_direction(self.frag_b)
            tip_a = self.tip_dir_a / torch.norm(self.tip_dir_a)
            cross = torch.linalg.cross(tip_b, tip_a)
            cos = torch.clamp(torch.dot(tip_b, tip_a), -1.0, 1.0).item()
            if torch.norm(cross) < 1e-6:
                axis = torch.linalg.cross(tip_a, torch.tensor([1.0, 0.0, 0.0], device=dev, dtype=dt))
                if torch.norm(axis) < 1e-6:
                    axis = torch.linalg.cross(tip_a, torch.tensor([0.0, 1.0, 0.0], device=dev, dtype=dt))
                R = self._axis_angle(axis, np.pi if cos < 0 else 0.0, dev, dt)
            else:
                R = self._axis_angle(cross, np.arccos(cos), dev, dt)

            # 2) roll attorno all'asse della punta (casuale se non indicato)
            if roll_deg is None:
                roll_deg = float(rng.uniform(0.0, 360.0))
            self.start_roll_deg = roll_deg
            R = self._axis_angle(tip_a, np.radians(roll_deg), dev, dt) @ R

            # 3) piccola inclinazione casuale (asse perpendicolare alla punta)
            perp = torch.as_tensor(rng.normal(size=3), device=dev, dtype=dt)
            perp = perp - torch.dot(perp, tip_a) * tip_a
            tilt = np.radians(rng.uniform(0.0, self.START_MAX_TILT))
            R = self._axis_angle(perp, tilt, dev, dt) @ R

            pts = pts @ R.T
            normals = normals @ R.T

            # 4) posizione: centro della maschera di B davanti al centro
            #    della maschera di A, staccato lungo la normale media di A
            mask_a = self.fracture_mask_a.bool()
            mask_b = self.fracture_mask_b.bool()
            center_frac_a = self.frag_a[mask_a].mean(dim=0) if torch.any(mask_a) else self.frag_a.mean(dim=0)
            center_frac_b = pts[mask_b].mean(dim=0) if torch.any(mask_b) else pts.mean(dim=0)
            mean_normal_a = self.frag_a_normals[mask_a].mean(dim=0) if torch.any(mask_a) else self.frag_a_normals.mean(dim=0)
            detach_dir = mean_normal_a / (torch.norm(mean_normal_a) + 1e-8)
            axial = self.principal_axis_a * float(rng.uniform(-self.START_MAX_AXIAL_SHIFT, self.START_MAX_AXIAL_SHIFT))

            self.frag_b = pts + (center_frac_a - center_frac_b) + detach_dir * self.START_DISTANCE + axial
            self.frag_b_normals = normals
            self._start_away_dir = detach_dir

    def _place_b_searched(self, rank):
        """Partenza dalla posa di search_start.py, staccata lungo la normale media della frattura di A."""
        dev, dt = self.device, self.frag_b.dtype
        with torch.no_grad():
            R = torch.as_tensor(self.start_poses["R"][rank], device=dev, dtype=dt)
            t = torch.as_tensor(self.start_poses["t"][rank], device=dev, dtype=dt)
            self.frag_b = self.frag_b_rest @ R.T + t
            self.frag_b_normals = self.frag_b_normals @ R.T
            mask_a = self.fracture_mask_a.bool()
            mean_normal_a = self.frag_a_normals[mask_a].mean(dim=0) if torch.any(mask_a) else self.frag_a_normals.mean(dim=0)
            detach_dir = mean_normal_a / (torch.norm(mean_normal_a) + 1e-8)
            self.frag_b = self.frag_b + detach_dir * self.START_DISTANCE
            self._start_away_dir = detach_dir
        self.episode_rank = rank

    def _potential(self, chamfer, normal_alignment, surface_matching_ratio, tip_alignment):
        """Qualita' della posa per il potential-based shaping (vedi __init__)."""
        return (
            self.W_TIP * tip_alignment
            + self.W_NORMAL * normal_alignment
            + self.W_COVERAGE * surface_matching_ratio
            - self.W_CHAMFER * min(chamfer, self.CHAMFER_POTENTIAL_CAP)
        )

    def _load_occupancy(self, file_name):
        """Griglia di occupazione del frammento (cache per nome file .pt)."""
        stem = os.path.splitext(os.path.basename(file_name))[0]
        if stem not in self._occupancy_cache:
            path = os.path.join(self.occupancy_dir, f"{stem}.npz")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Griglia di occupazione mancante: {path}. "
                    f"Generala con: python -m srcRL.occupancy"
                )
            self._occupancy_cache[stem] = OccupancyGrid.load(path, device=self.device)
        return self._occupancy_cache[stem]

    def _current_pose_b(self):
        """
        Rototraslazione (R, t) che porta B dalla posa del dataset
        (frag_b_rest, il frame della sua griglia di occupazione) alla
        posa corrente: frag_b = frag_b_rest @ R.T + t. Stimata con
        Kabsch, esatta perche' B subisce solo trasformazioni rigide.
        """
        p, q = self.frag_b_rest, self.frag_b
        pc, qc = p.mean(dim=0), q.mean(dim=0)
        u, _, vt = torch.linalg.svd((p - pc).T @ (q - qc))
        d = torch.sign(torch.det(vt.T @ u.T))
        D = torch.diag(torch.stack([torch.ones_like(d), torch.ones_like(d), d]))
        R = vt.T @ D @ u.T
        return R, qc - pc @ R.T

    def _detect_penetration(self):
        """
        Compenetrazione reale tra le ossa: punti di B dentro il materiale
        di A + punti di A dentro il materiale di B, letti dalle griglie di
        occupazione. Restituisce (num_punti_compenetrati, distanza minima
        B->A, frazione di punti compenetrati su A+B).
        """
        with torch.no_grad():
            dists = torch.cdist(self.frag_b, self.frag_a)
            min_dist_scalar = torch.min(dists).item()

            R, t = self._current_pose_b()
            a_in_b_frame = (self.frag_a - t) @ R  # inversa di x @ R.T + t

            b_inside_a = self.occupancy_a.inside(self.frag_b)
            a_inside_b = self.occupancy_b.inside(a_in_b_frame)

            num_contacts = int(b_inside_a.sum().item() + a_inside_b.sum().item())
            penetration_ratio = num_contacts / (self.frag_a.shape[0] + self.frag_b.shape[0])

            return num_contacts, min_dist_scalar, penetration_ratio

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        idx = np.random.randint(0, len(self.dataset))
        frag_a_full, frag_b_full, (graph_a, graph_b) = self.dataset[idx]
        file_a, file_b = self.dataset.pairs[idx]
        self.occupancy_a = self._load_occupancy(file_a)
        self.occupancy_b = self._load_occupancy(file_b)

        if hasattr(frag_a_full, 'x'):
            frag_a_full = frag_a_full.x
        if hasattr(frag_b_full, 'x'):
            frag_b_full = frag_b_full.x

        self.frag_a = (frag_a_full[:, :3] * 70).clone().detach().to(self.device)
        self.frag_b = (frag_b_full[:, :3] * 70).clone().detach().to(self.device)

        self.frag_a_normals = frag_a_full[:, 3:6].clone().detach().to(self.device)
        self.frag_b_normals = frag_b_full[:, 3:6].clone().detach().to(self.device)

        self.frag_a_normals = self._orient_normals_outward(self.frag_a, self.frag_a_normals)
        self.frag_b_normals = self._orient_normals_outward(self.frag_b, self.frag_b_normals)

        # Posa di B nel frame della sua griglia di occupazione (posa del
        # dataset), prima di qualsiasi trasformazione dell'episodio.
        self.frag_b_rest = self.frag_b.clone()

        if frag_a_full.shape[1] > 7:
            self.fracture_mask_a = (frag_a_full[:, 7] > 0.5)
            self.fracture_mask_b = (frag_b_full[:, 7] > 0.5)
        else:
            self.fracture_mask_a = torch.ones(frag_a_full.shape[0], dtype=torch.bool, device=self.device)
            self.fracture_mask_b = torch.ones(frag_b_full.shape[0], dtype=torch.bool, device=self.device)

        # Settori spaziali della fracture_mask di A, tramite K-Means a
        # K FISSO. A non si muove mai, quindi i settori sono stabili
        # per l'intero episodio.
        N_FRACTURE_SECTORS = 4

        frag_a_mask_points = self.frag_a[self.fracture_mask_a]
        if frag_a_mask_points.shape[0] >= N_FRACTURE_SECTORS:
            km = KMeans(n_clusters=N_FRACTURE_SECTORS, n_init=10, random_state=0).fit(
                frag_a_mask_points.detach().cpu().numpy()
            )
            labels = km.labels_
            self.fracture_cluster_labels_a = torch.tensor(labels, dtype=torch.long, device=self.device)
        else:
            self.fracture_cluster_labels_a = torch.zeros(
                frag_a_mask_points.shape[0], dtype=torch.long, device=self.device
            )

        # Asse principale di A (PCA sull'intero frammento). A non si
        # muove mai, quindi asse e centro sono stabili per l'intero
        # episodio. Usato per la penalita' morbida di sovrapposizione
        # assiale.
        centered_a = self.frag_a - self.frag_a.mean(dim=0)
        _, _, V_a = torch.linalg.svd(centered_a)
        self.principal_axis_a = V_a[0]
        self.axis_center_a = self.frag_a.mean(dim=0)
        self.principal_axes_a = V_a

        # 🆕 Direzione "punta" di A (fissa, A non si muove mai). Il
        # "polo" globale di orientamento -- vedi _tip_direction.
        self.tip_dir_a = self._tip_direction(self.frag_a)

        # DISABILITATO su richiesta: blocco che applicava T_GT +
        # scostamento di 15mm come punto di partenza per B.
        #
        '''
        T_GT = torch.tensor([
             [-0.264350, -0.115371,  0.957501,  32.685013],
             [ 0.196923, -0.978359, -0.063517, -44.966434],
             [ 0.944108,  0.171764,  0.281348,   4.563776],
             [ 0.000000,  0.000000,  0.000000,   1.000000]
         ], dtype=torch.float32, device=self.device)
        
        R_gt = T_GT[:3, :3]
        t_gt = T_GT[:3, 3]
        
        self.frag_b = (self.frag_b @ R_gt.T) + t_gt
        self.frag_b_normals = self.frag_b_normals @ R_gt.T
        
        mean_normal_a = self.frag_a_normals[self.fracture_mask_a].mean(dim=0)
        detach_dir = mean_normal_a / (torch.norm(mean_normal_a) + 1e-8)
        
        OFFSET_DIST = 15.0
        self.frag_b += detach_dir * OFFSET_DIST
        '''
        self._start_away_dir = None
        self.episode_face = None
        self.episode_rank = None
        if self.start_mode == "searched":
            rank = None if options is None else options.get("rank")
            self._place_b_searched(self.start_rank if rank is None else int(rank))
        elif self.start_mode in ("tip_aligned", "cube"):
            roll = None if options is None else options.get("roll_deg")
            self._roll_index = None
            if roll is None and self.roll_sampler is not None:
                self._roll_index, roll = self.roll_sampler.sample(self.np_random)
            self._place_b_tip_aligned(roll)
            if self.start_mode == "cube":
                face = None if options is None else options.get("face")
                face = face or self.start_face or str(self.np_random.choice(list(self.CUBE_FACES)))
                self._place_b_on_cube_face(face)
        else:
            # Rotazione CASUALE (diversa ogni episodio, genuinamente
            # uniforme su SO(3) -- vedi _random_rotation_matrix) intorno
            # al baricentro di B, al posto di T_GT.
            R_random = self._random_rotation_matrix(self.device, self.frag_b.dtype)

            centro_b_init = torch.mean(self.frag_b, dim=0)
            frag_b_centered_init = self.frag_b - centro_b_init
            transformed_init = torch.mm(frag_b_centered_init, R_random.t())

            self.frag_b = transformed_init + centro_b_init
            self.frag_b_normals = torch.mm(self.frag_b_normals, R_random.t())

        # La partenza non deve compenetrare: altrimenti ogni movimento
        # verrebbe annullato dal rollback e l'episodio resterebbe bloccato.
        # Si allontana B da A lungo la congiungente dei baricentri.
        with torch.no_grad():
            for _ in range(100):
                if self._detect_penetration()[2] <= self.PENETRATION_RATIO_TARGET:
                    break
                if self._start_away_dir is not None:
                    away = self._start_away_dir
                else:
                    away = self.frag_b.mean(dim=0) - self.frag_a.mean(dim=0)
                self.frag_b = self.frag_b + away / (torch.norm(away) + 1e-8) * 1.0

        if self.obs_mode == "legacy":
            with torch.no_grad():
                self.shape_emb = self._encode_pair(graph_a, graph_b).cpu().numpy()
        elif self.obs_mode == "interface":
            # Sottoinsiemi FISSI per l'episodio dei punti di frattura
            # usati come nodi del grafo di interfaccia: stessi punti
            # fisici ad ogni step, cambia solo la loro posa relativa.
            self.iface_idx_a = self._sample_iface_indices(self.fracture_mask_a)
            self.iface_idx_b = self._sample_iface_indices(self.fracture_mask_b)

        with torch.no_grad():
            current_chamfer, _, surface_matching_ratio, min_dist, _, normal_alignment, penetration_ratio, axial_overlap, tip_alignment = self._compute_geometric_metrics()

        self.current_step = 0
        self.prev_chamfer = current_chamfer
        self.prev_normal = normal_alignment
        self.prev_surface_matching_ratio = surface_matching_ratio
        self.prev_potential = self._potential(current_chamfer, normal_alignment, surface_matching_ratio, tip_alignment)
        self.episode_best_phi = (
            self.prev_potential if penetration_ratio <= self.PENETRATION_RATIO_TARGET else None
        )

        self.episode_best_chamfer = current_chamfer
        self.episode_best_normal = normal_alignment
        self.episode_best_surface_matching_ratio = surface_matching_ratio
        self.current_rotation = torch.zeros(3, device=self.frag_b.device, dtype=self.frag_b.dtype)

        if hasattr(self, 'center_a_frac'):
            delattr(self, 'center_a_frac')

        if self.best_frag_a is None:
            self.best_frag_a = self.frag_a.detach().cpu().numpy().copy()
            self.best_frag_b = self.frag_b.detach().cpu().numpy().copy()
            self.global_best_chamfer = current_chamfer
            self.global_best_normal = normal_alignment
            self.global_best_surface_matching_ratio = surface_matching_ratio
            self.global_best_penetration_ratio = penetration_ratio
            self.best_chamfer = current_chamfer
            self.best_normal = normal_alignment
            self.best_surface_matching_ratio = surface_matching_ratio
            self.best_penetration_ratio = penetration_ratio

        obs = self._get_obs(current_chamfer, min_dist, normal_alignment, tip_alignment)
        return obs, {}

    def step(self, action):
        self.current_step += 1

        device = self.frag_b.device
        dtype = self.frag_b.dtype

        posizione_precedente_b = self.frag_b.clone()
        normali_precedenti_b = self.frag_b_normals.clone()
        if isinstance(self.current_rotation, torch.Tensor):
            rotazione_precedente_b = self.current_rotation.clone()
        else:
            rotazione_precedente_b = self.current_rotation.copy()

        dx, dy, dz, d_roll, d_pitch, d_yaw = action

        step_scale = max(0.01, min(0.5, self.prev_chamfer * 0.03))        # mm
        rot_scale = max(0.002, min(0.05, self.prev_chamfer * 0.004))       # radianti

        trans = torch.tensor([dx, dy, dz], device=device, dtype=dtype) * step_scale
        delta_rot = torch.tensor([d_roll, d_pitch, d_yaw], device=device, dtype=dtype) * rot_scale

        with torch.no_grad():
            T = self._euler_and_translation_to_matrix(delta_rot, trans)
            centro_b = torch.mean(self.frag_b, dim=0)

            frag_b_centered = self.frag_b - centro_b
            ones = torch.ones((frag_b_centered.shape[0], 1), device=device, dtype=dtype)
            frag_b_homogeneous = torch.cat([frag_b_centered, ones], dim=1)

            transformed_homogeneous = torch.mm(frag_b_homogeneous, T.t())
            self.frag_b = transformed_homogeneous[:, :3] + centro_b

            R = T[:3, :3]
            self.frag_b_normals = torch.mm(self.frag_b_normals, R.t())
            self.current_rotation += delta_rot

            (
                current_chamfer,
                fracture_disparity,
                surface_matching_ratio,
                min_dist,
                num_contacts,
                normal_alignment,
                penetration_ratio,
                axial_overlap,
                tip_alignment,
            ) = self._compute_geometric_metrics()

        reward = 0.0
        terminated = False
        truncated = self.current_step >= self.max_steps
        is_success = False

        if current_chamfer > self.max_allowed_chamfer:
            reward = -20.0
            terminated = True
            self._end_episode()
            return (
                self._get_obs(current_chamfer, min_dist, normal_alignment, tip_alignment),
                float(reward),
                terminated,
                truncated,
                {
                    "chamfer": float(current_chamfer),
                    "contacts": 0,
                    "surface_matching_ratio": float(surface_matching_ratio),
                    "normal_alignment": float(normal_alignment),
                    "success": False,
                    **self._start_info(),
                },
            )

        if penetration_ratio > self.PENETRATION_ROLLBACK_RATIO:
            rollback_penalty = -1.0 - (penetration_ratio * self.PENETRATION_PENALTY_WEIGHT)
            if surface_matching_ratio < 0.50:
                mean_normal_a = self.frag_a_normals[self.fracture_mask_a].mean(dim=0)
                detach_dir = mean_normal_a / (torch.norm(mean_normal_a) + 1e-8)
                self.frag_b = posizione_precedente_b + (detach_dir * 0.8)
            else:
                self.frag_b = posizione_precedente_b
            self.frag_b_normals = normali_precedenti_b

            if isinstance(self.current_rotation, torch.Tensor):
                self.current_rotation = rotazione_precedente_b.clone()
            else:
                self.current_rotation = rotazione_precedente_b.copy()

            with torch.no_grad():
                (current_chamfer, _, surface_matching_ratio, min_dist, _, normal_alignment, penetration_ratio, axial_overlap, tip_alignment) = self._compute_geometric_metrics()
            info_contacts = num_contacts

            # La posa e' tornata (quasi) quella di prima: lo shaping e'
            # ~0, resta la penalita' per il tentativo di compenetrazione.
            potential = self._potential(current_chamfer, normal_alignment, surface_matching_ratio, tip_alignment)
            reward = (
                rollback_penalty
                + self.shaping_gamma * potential - self.prev_potential
                - self.STEP_PENALTY
            )
        else:
            info_contacts = num_contacts

            potential = self._potential(current_chamfer, normal_alignment, surface_matching_ratio, tip_alignment)
            reward_shaping = self.shaping_gamma * potential - self.prev_potential

            # Costi (mai positivi: non creano incentivi a restare fermi)
            penalty_penetration = -penetration_ratio * self.PENETRATION_PENALTY_WEIGHT
            penalty_axial_overlap = -max(0.0, axial_overlap - self.AXIAL_OVERLAP_SOFT_THRESHOLD) * 20.0

            reward = reward_shaping + penalty_penetration + penalty_axial_overlap - self.STEP_PENALTY

            if current_chamfer < self.episode_best_chamfer:
                self.episode_best_chamfer = current_chamfer

            if normal_alignment > self.episode_best_normal:
                self.episode_best_normal = normal_alignment

            if surface_matching_ratio > self.episode_best_surface_matching_ratio:
                self.episode_best_surface_matching_ratio = surface_matching_ratio

            if self.global_best_chamfer < self.CHAMFER_GOOD_ENOUGH:
                if (
                penetration_ratio <= self.PENETRATION_RATIO_TARGET
                and normal_alignment >= self.NORMAL_ALIGNMENT_TARGET
                and surface_matching_ratio >= self.global_best_surface_matching_ratio
                and min_dist <= self.CONTACT_MIN_DIST_TARGET
                and tip_alignment >= self.TIP_ALIGNMENT_TARGET
                and current_chamfer <= self.CHAMFER_GOOD_ENOUGH
                ):
                    self.global_best_chamfer = current_chamfer
                    self.global_best_normal = normal_alignment
                    self.global_best_surface_matching_ratio = surface_matching_ratio
                    self.global_best_penetration_ratio = penetration_ratio
                    self.best_chamfer = current_chamfer
                    self.best_normal = normal_alignment
                    self.best_surface_matching_ratio = surface_matching_ratio
                    self.best_penetration_ratio = penetration_ratio
                    self.best_frag_a = self.frag_a.detach().cpu().numpy().copy()
                    self.best_frag_b = self.frag_b.detach().cpu().numpy().copy()
            elif (
                penetration_ratio <= self.PENETRATION_RATIO_TARGET
                and normal_alignment >= self.NORMAL_ALIGNMENT_TARGET
                and surface_matching_ratio >= self.global_best_surface_matching_ratio
                and min_dist <= self.CONTACT_MIN_DIST_TARGET
                and tip_alignment >= self.TIP_ALIGNMENT_TARGET
                ):
                    self.global_best_chamfer = current_chamfer
                    self.global_best_normal = normal_alignment
                    self.global_best_surface_matching_ratio = surface_matching_ratio
                    self.global_best_penetration_ratio = penetration_ratio
                    self.best_chamfer = current_chamfer
                    self.best_normal = normal_alignment
                    self.best_surface_matching_ratio = surface_matching_ratio
                    self.best_penetration_ratio = penetration_ratio
                    self.best_frag_a = self.frag_a.detach().cpu().numpy().copy()
                    self.best_frag_b = self.frag_b.detach().cpu().numpy().copy()


            if (penetration_ratio <= self.PENETRATION_RATIO_TARGET
                and normal_alignment >= self.NORMAL_ALIGNMENT_TARGET
                and min_dist <= self.CONTACT_MIN_DIST_TARGET
                and surface_matching_ratio >= self.SURFACE_MATCHING_TARGET
                and tip_alignment >= self.TIP_ALIGNMENT_TARGET
                and current_chamfer <= self.CHAMFER_GOOD_ENOUGH):
                reward += self.SUCCESS_REWARD
                terminated = True
                is_success = True
                print(
                    f"🎯 INCASTRO TROVATO AL PASSO {self.current_step}! "
                    f"(Chamfer: {current_chamfer:.4f} mm, Norm Alignment: {normal_alignment:.4f}, "
                    f"Coverage: {surface_matching_ratio:.2f}, Tip: {tip_alignment:.2f})"
                )

        self.prev_chamfer = current_chamfer
        self.prev_normal = normal_alignment
        self.prev_surface_matching_ratio = surface_matching_ratio
        self.prev_potential = potential
        if penetration_ratio <= self.PENETRATION_RATIO_TARGET and (
            self.episode_best_phi is None or potential > self.episode_best_phi
        ):
            self.episode_best_phi = potential
        obs = self._get_obs(current_chamfer, min_dist, normal_alignment, tip_alignment)
        if terminated or truncated:
            self._end_episode()

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            {
                "chamfer": float(current_chamfer),
                "contacts": info_contacts,
                "surface_matching_ratio": float(surface_matching_ratio),
                "normal_alignment": float(normal_alignment),
                "penetration_ratio": float(penetration_ratio),
                "tip_alignment": float(tip_alignment),
                "success": is_success,
                **self._start_info(),
            },
        )

    def _compute_geometric_metrics(self):
        with torch.no_grad():
            mask_a_bool = self.fracture_mask_a.bool()
            mask_b_bool = self.fracture_mask_b.bool()

            frac_a = self.frag_a[mask_a_bool] if torch.any(mask_a_bool) else self.frag_a
            frac_b = self.frag_b[mask_b_bool] if torch.any(mask_b_bool) else self.frag_b

            norm_a_surf = self.frag_a_normals[mask_a_bool] if torch.any(mask_a_bool) else self.frag_a_normals
            norm_b_surf = self.frag_b_normals[mask_b_bool] if torch.any(mask_b_bool) else self.frag_b_normals

            dists_frac = torch.cdist(frac_b, frac_a)
            min_b2a_frac = torch.min(dists_frac, dim=1).values
            min_a2b_frac = torch.min(dists_frac, dim=0).values

            CHAMFER_KEEP_FRACTION = 0.5

            k_b = max(1, int(len(min_b2a_frac) * CHAMFER_KEEP_FRACTION))
            k_a = max(1, int(len(min_a2b_frac) * CHAMFER_KEEP_FRACTION))

            trimmed_b2a = torch.topk(min_b2a_frac, k_b, largest=False).values.mean()
            trimmed_a2b = torch.topk(min_a2b_frac, k_a, largest=False).values.mean()

            current_chamfer = (trimmed_b2a + trimmed_a2b).item()

            idx_b2a = torch.argmin(dists_frac, dim=1)
            corresponding_normal_a = norm_a_surf[idx_b2a]

            local_dot = torch.sum(norm_b_surf * corresponding_normal_a, dim=1)
            local_alignment = 0.5 * (1.0 - local_dot)

            proximity_scale = max(self.collision_threshold, 1e-3)
            weight = torch.exp(-min_b2a_frac / proximity_scale)
            weight_sum = weight.sum()

            if weight_sum > 1e-6:
                normal_alignment = (local_alignment * weight).sum().item() / weight_sum.item()
            else:
                normal_alignment = 0.0

            num_contacts, min_dist_scalar, penetration_ratio = self._detect_penetration()

            proximity_band = 2.0  # mm
            matching_b = (torch.sum(min_b2a_frac < proximity_band).float() / min_b2a_frac.numel()).item()

            cluster_labels = self.fracture_cluster_labels_a
            unique_clusters = torch.unique(cluster_labels)
            cluster_ratios = []

            for c in unique_clusters:
                if c.item() == -1:
                    continue
                idx = (cluster_labels == c)
                if idx.sum() == 0:
                    continue
                frac_close = (min_a2b_frac[idx] < proximity_band).float().mean()
                cluster_ratios.append(frac_close)

            if len(cluster_ratios) > 0:
                cluster_tensor = torch.stack(cluster_ratios)
                cluster_mean = cluster_tensor.mean().item()
                cluster_min = cluster_tensor.min().item()
                matching_a = 0.6 * cluster_mean + 0.4 * cluster_min
            else:
                matching_a = 0.0

            surface_matching_ratio = 0.5 * (matching_b + matching_a)

            max_gap = torch.max(min_b2a_frac).item()
            min_gap = torch.min(min_b2a_frac).item()
            fracture_disparity = max_gap - min_gap

            # Sovrapposizione assiale (asse principale di A, fisso).
            proj_a = torch.matmul(self.frag_a - self.axis_center_a, self.principal_axis_a)
            proj_b = torch.matmul(self.frag_b - self.axis_center_a, self.principal_axis_a)

            a_min, a_max = proj_a.min().item(), proj_a.max().item()
            b_min, b_max = proj_b.min().item(), proj_b.max().item()

            axial_overlap_raw = max(0.0, min(a_max, b_max) - max(a_min, b_min))
            a_len = max(a_max - a_min, 1e-6)
            b_len = max(b_max - b_min, 1e-6)
            axial_overlap = axial_overlap_raw / min(a_len, b_len)

            # 🆕 "Polo": dot tra direzione punta di A (fissa) e di B
            # (ricalcolata ogni volta dalla posa corrente di frag_b --
            # sotto trasformazione rigida il punto piu' lontano dal
            # baricentro resta lo stesso punto fisico, quindi la
            # direzione ruota correttamente insieme al frammento).
            tip_dir_b = self._tip_direction(self.frag_b)
            tip_alignment = torch.dot(self.tip_dir_a, tip_dir_b).item()

            return (
                current_chamfer, fracture_disparity, surface_matching_ratio, min_dist_scalar,
                num_contacts, normal_alignment, penetration_ratio, axial_overlap, tip_alignment,
            )

    def _get_obs(self, current_chamfer, min_dist, normal_alignment, tip_alignment):
        with torch.no_grad():
            if not hasattr(self, "center_a_frac"):
                mask_a_bool = self.fracture_mask_a.bool()
                self.center_a_frac = (
                    torch.mean(self.frag_a[mask_a_bool], dim=0)
                    if torch.any(mask_a_bool)
                    else torch.mean(self.frag_a, dim=0)
                )

            mask_b_bool = self.fracture_mask_b.bool()
            center_b_frac = (
                torch.mean(self.frag_b[mask_b_bool], dim=0)
                if torch.any(mask_b_bool)
                else torch.mean(self.frag_b, dim=0)
            )

            approach_vector = self.center_a_frac - center_b_frac
            approach_dir = (
                approach_vector / (torch.norm(approach_vector) + 1e-8)
            ).cpu().numpy()

            if torch.any(mask_b_bool):
                mean_norm_b = torch.mean(self.frag_b_normals[mask_b_bool], dim=0)
                mean_norm_b_dir = (
                    mean_norm_b / (torch.norm(mean_norm_b) + 1e-8)
                ).cpu().numpy()
            else:
                mean_norm_b_dir = np.zeros(3, dtype=np.float32)

        if isinstance(self.current_rotation, torch.Tensor):
            r, p, y = self.current_rotation.cpu().numpy()
        else:
            r, p, y = self.current_rotation

        geom_obs = np.array(
            [
                float(current_chamfer),
                float(normal_alignment),
                float(min_dist),
                np.sin(r),
                np.cos(r),
                np.sin(p),
                np.cos(p),
                np.sin(y),
                np.cos(y),
                approach_dir[0],
                approach_dir[1],
                approach_dir[2],
                mean_norm_b_dir[0],
                mean_norm_b_dir[1],
                mean_norm_b_dir[2],
                float(tip_alignment),  # 🆕 il "polo": segnale globale di orientamento
            ],
            dtype=np.float32,
        )

        if self.obs_mode == "legacy":
            return np.concatenate([geom_obs, self.shape_emb]).astype(np.float32)
        if self.obs_mode == "geom":
            return geom_obs

        return {"geom": geom_obs, "iface": self._build_interface_features()}

    def _encode_pair(self, graph_a, graph_b):
        """Embedding globale (legacy) della coppia nella posa del dataset."""
        emb_a = self.encoder(graph_a.to(self.device))
        emb_b = self.encoder(graph_b.to(self.device))
        return torch.cat([emb_a, emb_b], dim=-1).flatten()

    def _sample_iface_indices(self, fracture_mask):
        """
        Indici (sull'intero frammento) di n_iface_points punti della
        maschera di frattura. Se la maschera e' vuota si usa l'intero
        frammento, come in _compute_geometric_metrics; se ha meno punti
        del necessario si campiona con ripetizione.
        """
        candidates = torch.nonzero(fracture_mask.bool(), as_tuple=False).flatten()
        if candidates.numel() == 0:
            candidates = torch.arange(fracture_mask.shape[0], device=fracture_mask.device)
        replace = candidates.numel() < self.n_iface_points
        choice = self.np_random.choice(candidates.numel(), size=self.n_iface_points, replace=replace)
        return candidates[torch.as_tensor(choice, device=candidates.device)]

    def _interface_side_features(self, pts, normals, other_pts, other_normals, side_value, center):
        """
        Feature dei nodi di un lato dell'interfaccia, rispetto alla
        superficie di frattura dell'altro frammento:
          0-2  posizione relativa al centro frattura di A
          3-5  normale
          6    lato (0 = A, 1 = B)
          7    distanza dal punto piu' vicino dell'altro frammento
          8    dot tra la normale e quella del punto piu' vicino
               (-1 = superfici affacciate, come all'incastro)
          9    distanza con segno lungo la normale dell'altro punto
               (negativa = compenetrazione)
          10-12 vettore verso il punto piu' vicino (direzione in cui
               il nodo dovrebbe spostarsi per chiudere il gap)
        Le distanze sono compresse con tanh per restare limitate.
        """
        dists = torch.cdist(pts, other_pts)
        min_d, nn_idx = torch.min(dists, dim=1)
        nn_pts = other_pts[nn_idx]
        nn_normals = other_normals[nn_idx]

        to_nn = nn_pts - pts
        signed = torch.sum((pts - nn_pts) * nn_normals, dim=1)
        normal_dot = torch.sum(normals * nn_normals, dim=1)
        side = torch.full_like(min_d, float(side_value))

        ds = self.IFACE_DIST_SCALE
        return torch.cat([
            (pts - center) / self.IFACE_POS_SCALE,
            normals,
            side.unsqueeze(1),
            torch.tanh(min_d / ds).unsqueeze(1),
            normal_dot.unsqueeze(1),
            torch.tanh(signed / ds).unsqueeze(1),
            torch.tanh(to_nn / ds),
        ], dim=1)

    def _build_interface_features(self):
        """Nodi del grafo di interfaccia A-B nella posa corrente: (2*n_iface_points, 13)."""
        with torch.no_grad():
            mask_a_bool = self.fracture_mask_a.bool()
            mask_b_bool = self.fracture_mask_b.bool()
            frac_a = self.frag_a[mask_a_bool] if torch.any(mask_a_bool) else self.frag_a
            frac_b = self.frag_b[mask_b_bool] if torch.any(mask_b_bool) else self.frag_b
            norm_a = self.frag_a_normals[mask_a_bool] if torch.any(mask_a_bool) else self.frag_a_normals
            norm_b = self.frag_b_normals[mask_b_bool] if torch.any(mask_b_bool) else self.frag_b_normals

            center = self.center_a_frac

            feat_a = self._interface_side_features(
                self.frag_a[self.iface_idx_a], self.frag_a_normals[self.iface_idx_a],
                frac_b, norm_b, 0.0, center,
            )
            feat_b = self._interface_side_features(
                self.frag_b[self.iface_idx_b], self.frag_b_normals[self.iface_idx_b],
                frac_a, norm_a, 1.0, center,
            )
            return torch.cat([feat_a, feat_b], dim=0).cpu().numpy().astype(np.float32)

    def _euler_and_translation_to_matrix(self, angles, translation):
        r, p, y = angles
        tx, ty, tz = translation
        device, dtype = self.frag_b.device, self.frag_b.dtype

        r_t, p_t, y_t = torch.as_tensor(r, device=device, dtype=dtype), torch.as_tensor(p, device=device, dtype=dtype), torch.as_tensor(y, device=device, dtype=dtype)
        cos_r, sin_r = torch.cos(r_t), torch.sin(r_t)
        cos_p, sin_p = torch.cos(p_t), torch.sin(p_t)
        cos_y, sin_y = torch.cos(y_t), torch.sin(y_t)

        Rx = torch.eye(3, device=device, dtype=dtype)
        Rx[1, 1], Rx[1, 2] = cos_r, -sin_r
        Rx[2, 1], Rx[2, 2] = sin_r, cos_r

        Ry = torch.eye(3, device=device, dtype=dtype)
        Ry[0, 0], Ry[0, 2] = cos_p, sin_p
        Ry[2, 0], Ry[2, 2] = -sin_p, cos_p

        Rz = torch.eye(3, device=device, dtype=dtype)
        Rz[0, 0], Rz[0, 1] = cos_y, -sin_y
        Rz[1, 0], Rz[1, 1] = sin_y, cos_y

        R = torch.mm(Rz, torch.mm(Ry, Rx))
        T = torch.eye(4, device=device, dtype=dtype)
        T[:3, :3] = R
        T[0, 3], T[1, 3], T[2, 3] = torch.as_tensor(tx, device=device, dtype=dtype), torch.as_tensor(ty, device=device, dtype=dtype), torch.as_tensor(tz, device=device, dtype=dtype)

        return T
